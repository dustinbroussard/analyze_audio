#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal, fft as spfft

# Optional imports
HAVE_LIBROSA = True
try:
    import librosa
except Exception:
    HAVE_LIBROSA = False

@dataclass
class AnalysisResult:
    path: str
    duration_sec: float
    sample_rate: int
    rms: float
    spectral_centroid_hz_avg: float
    tempo_bpm: Optional[float]
    tempo_confidence: Optional[float]
    key_guess: Optional[str]
    mode_guess: Optional[str]
    notes: str
    chord_segments: Optional[list] = None  # list of {start_sec, end_sec, label, confidence}

KEY_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def safe_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = np.max(np.abs(x)) if x.size else 0.0
    if not np.isfinite(m) or m < eps:
        return x.astype(np.float32, copy=True)
    return (x / m).astype(np.float32, copy=False)

def read_audio_mono(path: str, normalize: bool = True) -> Tuple[np.ndarray, int]:
    """
    Try soundfile first; if it fails on mp3/ogg (libsndfile build),
    fall back to librosa if available.
    """
    try:
        y, sr = sf.read(path, always_2d=False)
    except Exception:
        if not HAVE_LIBROSA:
            print(f"Could not read '{path}' with soundfile and librosa is unavailable.", file=sys.stderr)
            sys.exit(2)
        y, sr = librosa.load(path, sr=None, mono=True)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    if normalize:
        y = safe_norm(y)
    # NaN/Inf guard
    y[~np.isfinite(y)] = 0.0
    return y, sr

def hp_filter(x: np.ndarray, sr: int, cutoff_hz: float = 60.0) -> np.ndarray:
    # 4th-order Butter HPF, zero-phase
    b, a = signal.butter(4, cutoff_hz / (sr * 0.5), btype='highpass')
    return signal.filtfilt(b, a, x, method='gust')

def autocorr_fft(x: np.ndarray) -> np.ndarray:
    # real autocorr via FFT (power spectrum -> IFFT)
    n = int(2 ** np.ceil(np.log2(len(x) * 2 - 1)))
    X = spfft.rfft(x, n=n)
    ac = spfft.irfft(np.abs(X) ** 2, n=n)
    ac = ac[:len(x)]
    # normalize so ac[0] == 1 if possible
    if ac[0] > 0:
        ac = ac / (ac[0] + 1e-12)
    return ac

def estimate_tempo_hybrid(y: np.ndarray, sr: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Hybrid tempo estimate with minimal CPU:
    1) Onset proxy from HP energy envelope.
    2) Downsample envelope to ~200 Hz for stability.
    3) FFT autocorr + peak pick within 40–200 BPM.
    Returns (bpm, confidence in ~0..1+).
    """
    # Downsample audio to a moderate rate for envelope calc
    target_sr = 11025
    if sr != target_sr:
        # polyphase resampling reduces aliasing compared to decimate
        y_ds = signal.resample_poly(y, up=target_sr, down=sr)
        sr_ds = target_sr
    else:
        y_ds, sr_ds = y, sr

    y_hp = hp_filter(y_ds, sr_ds, cutoff_hz=60.0)
    onset_env = np.abs(y_hp)
    # Light median smoothing; kernel must be odd
    onset_env = signal.medfilt(onset_env, kernel_size=5)
    onset_env -= np.median(onset_env)
    onset_env[onset_env < 0] = 0.0

    # Frame to ~200 Hz
    hop = max(1, int(sr_ds / 200))
    framed = onset_env[::hop].astype(np.float32)
    if framed.size < 128 or np.allclose(framed, 0):
        return None, None
    framed -= framed.mean()
    std = np.std(framed)
    if std > 1e-8:
        framed /= std

    ac = autocorr_fft(framed)
    fps = sr_ds / hop
    min_bpm, max_bpm = 40.0, 200.0
    min_lag = int(np.floor(fps * 60.0 / max_bpm))
    max_lag = int(np.ceil (fps * 60.0 / min_bpm))
    min_lag = max(min_lag, 1)
    max_lag = min(max_lag, len(ac) - 1)
    if max_lag <= min_lag + 3:
        return None, None

    roi = ac[min_lag:max_lag]
    if roi.size < 8:
        return None, None

    # Peak picking with basic separation
    distance = max(1, int(fps * 60.0 / max_bpm))
    peaks, _ = signal.find_peaks(roi, distance=distance)
    if peaks.size == 0:
        return None, None

    # Choose best peak by height
    best_idx = int(peaks[np.argmax(roi[peaks])])
    best_lag = min_lag + best_idx
    raw_bpm = 60.0 * fps / best_lag

    # Half/double adjustment to human-groove window
    candidates = [raw_bpm, raw_bpm * 2, raw_bpm / 2]
    # Prefer 70–170 with minimal distance
    def score_bpm(b):
        target = 110.0
        penalty = 0.0
        if not (40 <= b <= 220):
            penalty += 10.0
        if 70 <= b <= 170:
            penalty -= 2.0
        return abs(b - target) + penalty
    bpm = min(candidates, key=score_bpm)

    # Confidence: peak prominence vs local mean (clamped to ~[0,1.5])
    neighborhood = roi[max(0, best_idx - 8): best_idx + 9]
    local = float(roi[best_idx])
    baseline = float(np.mean(neighborhood) + 1e-6)
    conf = max(0.0, min(1.5, local / baseline))

    return float(bpm), float(conf)

def spectral_centroid_avg(y: np.ndarray, sr: int) -> float:
    # STFT magnitude centroid averaged over frames
    n_fft = 2048
    hop = 512
    win = signal.windows.hann(n_fft, sym=False)
    # Pad for last frame
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    frames = []
    for i in range(0, len(y) - n_fft + 1, hop):
        seg = y[i:i + n_fft] * win
        mag = np.abs(spfft.rfft(seg))
        frames.append(mag)
    if not frames:
        return 0.0
    M = np.stack(frames, axis=1)  # (freq, time)
    freqs = np.linspace(0, sr / 2, M.shape[0])
    S = M + 1e-12
    centroid = np.sum(freqs[:, None] * S, axis=0) / np.sum(S, axis=0)
    return float(np.mean(centroid))

def estimate_key_librosa(y: np.ndarray, sr: int) -> Tuple[str, str]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma /= (np.max(chroma, axis=0, keepdims=True) + 1e-9)
    chroma_mean = np.mean(chroma, axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    scores = []
    for i in range(12):
        scores.append((i, 'major', float(np.dot(np.roll(major_profile, i), chroma_mean))))
        scores.append((i, 'minor', float(np.dot(np.roll(minor_profile, i), chroma_mean))))
    best = max(scores, key=lambda x: x[2])
    return KEY_NAMES[best[0]], best[1]

def estimate_key_fallback(y: np.ndarray, sr: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Lightweight STFT-chroma + Krumhansl without librosa.
    Crude but serviceable; returns (key, mode) or (None, None) if unstable.
    """
    n_fft = 4096
    hop = 1024
    win = signal.windows.hann(n_fft, sym=False)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    mags = []
    for i in range(0, len(y) - n_fft + 1, hop):
        seg = y[i:i + n_fft] * win
        mag = np.abs(spfft.rfft(seg))
        mags.append(mag)
    if not mags:
        return None, None
    M = np.stack(mags, axis=1)  # (freq, time)
    freqs = np.linspace(0, sr/2, M.shape[0])
    # Map freq bins to pitch classes
    with np.errstate(divide='ignore'):
        midi = 69 + 12 * np.log2((freqs + 1e-12) / 440.0)
    pitch_class = np.mod(np.round(midi).astype(int), 12)
    chroma = np.zeros((12, M.shape[1]), dtype=np.float32)
    for k in range(12):
        chroma[k] = M[pitch_class == k].sum(axis=0) if np.any(pitch_class == k) else 0.0
    chroma /= (np.max(chroma, axis=0, keepdims=True) + 1e-9)
    chroma_mean = np.mean(chroma, axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    scores = []
    for i in range(12):
        scores.append((i, 'major', float(np.dot(np.roll(major_profile, i), chroma_mean))))
        scores.append((i, 'minor', float(np.dot(np.roll(minor_profile, i), chroma_mean))))
    best = max(scores, key=lambda x: x[2])
    return KEY_NAMES[best[0]], best[1]

def rms_energy(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))

def write_click_wav(out_path: str, sr: int, duration_sec: float, bpm: float, strong_every=4):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    n = int(sr * duration_sec)
    audio = np.zeros(n, dtype=np.float32)
    spb = 60.0 / bpm
    n_beats = max(1, int(duration_sec / spb) + 1)
    click_len = int(0.02 * sr)  # 20 ms
    for b in range(n_beats):
        start = int(b * spb * sr)
        end = min(start + click_len, n)
        if start >= n:
            break
        freq = 2000.0 if (b % strong_every == 0) else 1000.0
        t = np.arange(end - start) / sr
        click = 0.8 * np.hanning(end - start) * np.sin(2*np.pi*freq*t)
        audio[start:end] += click.astype(np.float32)
    m = np.max(np.abs(audio))
    if m > 0:
        audio = (audio / m * 0.9).astype(np.float32)
    sf.write(out_path, audio, sr)

def _chord_templates_major_minor(include_sevenths: bool = True) -> Tuple[np.ndarray, list]:
    """
    Returns (templates, labels) for 24 major/minor triads across 12 pitch classes.
    templates: shape (24, 12), L2-normalized.
    labels: list like ["C:maj", "C:min", ...]
    """
    templates = []
    labels = []
    # triads
    triads = {
        'maj': [0, 4, 7],
        'min': [0, 3, 7],
    }
    sevenths = {
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10],
        '7':    [0, 4, 7, 10],  # dominant 7th
    }
    for root in range(12):
        for qual, pat in triads.items():
            t = np.zeros(12, dtype=np.float32)
            for p in pat:
                t[(root + p) % 12] = 1.0
            t /= (np.linalg.norm(t) + 1e-9)
            templates.append(t)
            labels.append(f"{KEY_NAMES[root]}:{qual}")
        if include_sevenths:
            for qual, pat in sevenths.items():
                t = np.zeros(12, dtype=np.float32)
                for p in pat:
                    t[(root + p) % 12] = 1.0
                t /= (np.linalg.norm(t) + 1e-9)
                templates.append(t)
                labels.append(f"{KEY_NAMES[root]}:{qual}")
    return np.stack(templates, axis=0), labels

def estimate_chords_librosa(y: np.ndarray, sr: int,
                            hop_length: int = 512,
                            min_seg_dur: float = 0.3,
                            confidence_threshold: float = 0.3,
                            include_sevenths: bool = True,
                            key_hint: Optional[Tuple[Optional[str], Optional[str]]] = None,
                            key_bias_strength: float = 0.05) -> list:
    """
    Practical chord detection using chroma + template matching + smoothing.
    Returns a list of segments [{start_sec, end_sec, label, confidence}].
    Requires librosa.
    """
    if not HAVE_LIBROSA:
        return []
    try:
        # robust chroma using CQT; fallback to chroma_stft if CQT fails
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        except Exception:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        # normalize frames
        chroma /= (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
        energy = chroma.sum(axis=0)
        templates, labels = _chord_templates_major_minor(include_sevenths=include_sevenths)
        # scores: (24, T)
        scores = templates @ chroma
        # Key-aware bias: reward diatonic chords for detected key/mode
        if key_hint and key_hint[0] and key_hint[1]:
            keyname, mode = key_hint
            try:
                root_index = KEY_NAMES.index(keyname)
            except Exception:
                root_index = None
            if root_index is not None:
                # basic diatonic set for major/minor
                # for major: I maj/maj7, ii min/min7, iii min/min7, IV maj/maj7, V maj/7 (dominant), vi min/min7
                # for minor (natural-ish): i min/min7, III maj/maj7, iv min/min7, v min/min7 (allow V7), VI maj/maj7, VII maj/7
                diatonic = set()
                if mode.lower().startswith('maj'):
                    degrees = [0, 2, 4, 5, 7, 9]
                    quals = {
                        0: ('maj', 'maj7'), 2: ('min', 'min7'), 4: ('min', 'min7'),
                        5: ('maj', 'maj7'), 7: ('maj', '7'), 9: ('min', 'min7')
                    }
                else:
                    degrees = [0, 3, 5, 7, 8, 10]
                    quals = {
                        0: ('min', 'min7'), 3: ('maj', 'maj7'), 5: ('min', 'min7'),
                        7: ('min', 'min7', '7'), 8: ('maj', 'maj7'), 10: ('maj', '7')
                    }
                for d in degrees:
                    pc = (root_index + d) % 12
                    for qset in (quals.get(d, ()),):
                        for q in qset:
                            diatonic.add(f"{KEY_NAMES[pc]}:{q}")
                # apply bias
                bias = np.zeros(len(labels), dtype=np.float32)
                for i, lab in enumerate(labels):
                    if lab in diatonic:
                        bias[i] = key_bias_strength
                scores = scores + bias[:, None]
        best_idx = np.argmax(scores, axis=0)
        best_conf = scores[best_idx, np.arange(scores.shape[1])]
        # energy gate -> "N" label where too low
        thr = np.percentile(energy, 20)
        labels_seq = [labels[idx] if energy[t] >= max(1e-6, thr) else "N" for t, idx in enumerate(best_idx)]
        conf_seq = [float(best_conf[t]) if labels_seq[t] != "N" else 0.0 for t in range(len(best_idx))]
        # median filter to stabilize
        k = 9 if len(labels_seq) >= 9 else (len(labels_seq)//2*2 + 1)
        if k >= 3:
            # map labels to ints for filtering (N=-1)
            label_to_int = {lab: i for i, lab in enumerate(labels)}
            ints = np.array([label_to_int.get(l, -1) for l in labels_seq])
            ints_f = signal.medfilt(ints, kernel_size=k)
            labels_seq = [labels[i] if i >= 0 else "N" for i in ints_f]
        times = librosa.frames_to_time(np.arange(len(labels_seq)+1), sr=sr, hop_length=hop_length)
        # collect segments
        segs = []
        start_f = 0
        for f in range(1, len(labels_seq)+1):
            if f == len(labels_seq) or labels_seq[f] != labels_seq[start_f]:
                label = labels_seq[start_f]
                s, e = float(times[start_f]), float(times[f])
                dur = e - s
                if label != "N" and dur >= min_seg_dur:
                    conf = float(np.mean(conf_seq[start_f:f]))
                    if conf >= confidence_threshold:
                        segs.append({"start_sec": s, "end_sec": e, "label": label, "confidence": conf})
                start_f = f
        return segs
    except Exception:
        return []

def write_chords_json(path: str, segments: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2)

def _fmt_time(t: float) -> str:
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:06.3f}"

def write_chords_txt(path: str, segments: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for seg in segments:
            f.write(f"{_fmt_time(seg['start_sec'])} - {_fmt_time(seg['end_sec'])}: {seg['label']} (conf={seg['confidence']:.2f})\n")

def write_chords_chordpro(path: str, segments: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("{title: Chord Timeline}\n")
        for seg in segments:
            f.write(f"[{seg['label']}] {_fmt_time(seg['start_sec'])}-{_fmt_time(seg['end_sec'])}\n")

def extract_best_loop(y: np.ndarray, sr: int, bpm: float, bars: int = 4, bars_to_scan: int = 32, beats_per_bar: int = 4):
    """
    Improved naive loop finder:
    - assume fixed meter (beats_per_bar)
    - score start candidates at *beat boundaries*
    - similarity = cosine(head, tail); penalize crest; short crossfade external
    Returns (start, end) in samples or (None, None).
    """
    if not bpm or bpm <= 0:
        return None, None
    spb = 60.0 / bpm
    spp = spb * beats_per_bar
    seg_len = int(bars * spp * sr)

    # Estimate how many complete bars exist
    total_bars = int((len(y) / sr) / spp)
    scan_bars = min(max(0, total_bars - bars), bars_to_scan)
    if scan_bars <= 0:
        return None, None

    # Build beat-aligned candidate starts
    starts = (np.arange(scan_bars) * spp * sr).astype(int)

    best_score, best_start = -1e9, None
    win = int(0.2 * sr)  # 200 ms compare windows
    for start in starts:
        end = start + seg_len
        if end + win >= len(y):
            break
        seg = y[start:end]
        head = seg[:win]
        tail = seg[-win:]
        # cosine similarity robust to scale
        h = np.dot(head, head) ** 0.5 + 1e-9
        t = np.dot(tail, tail) ** 0.5 + 1e-9
        cos = float(np.dot(head, tail) / (h * t))
        crest = float(np.max(np.abs(seg)) / (np.sqrt(np.mean(seg**2) + 1e-12) + 1e-12))
        score = cos - 0.12 * (crest - 1.0)
        if np.isnan(score):
            continue
        if score > best_score:
            best_score, best_start = score, start

    if best_start is None:
        return None, None
    return best_start, best_start + seg_len

def main():
    ap = argparse.ArgumentParser(description="CPU-friendly audio groove analyzer (polished)")
    ap.add_argument("path", help="Path to audio file (mp3/wav/flac/ogg)")
    ap.add_argument("--export-click", metavar="CLICK_WAV", help="Write a metronome click WAV")
    ap.add_argument("--export-loop", metavar="LOOP_WAV", help="Write a loop WAV")
    ap.add_argument("--loop-bars", type=int, default=4, help="Bars to include in exported loop (default 4)")
    ap.add_argument("--bars-to-scan", type=int, default=32, help="Bars to scan from the start when seeking best loop")
    ap.add_argument("--beats-per-bar", type=int, default=4, help="Time signature numerator (default 4)")
    ap.add_argument("--force-bpm", type=float, default=None, help="Override/force BPM if detector fails")
    ap.add_argument("--no-normalize", action="store_true", help="Disable input peak normalization")
    ap.add_argument("--export-json", metavar="RESULT_JSON", help="Also write JSON result to this path")
    ap.add_argument("--click-sr", type=int, default=None, help="Sample rate for click export (defaults to source sr)")
    ap.add_argument("--no-chords", action="store_true", help="Disable chord detection (saves CPU)")
    ap.add_argument("--export-chords-json", metavar="CHORDS_JSON", help="Write chord segments as JSON")
    ap.add_argument("--export-chords-txt", metavar="CHORDS_TXT", help="Write chord segments as plain text")
    ap.add_argument("--export-chords-chordpro", metavar="CHORDS_CHORDPRO", help="Write chord segments in ChordPro style")
    ap.add_argument("--chord-confidence", type=float, default=0.3, help="Chord segment min confidence (0-1)")
    ap.add_argument("--no-chord-key-bias", action="store_true", help="Disable key-aware chord bias")
    ap.add_argument("--no-chord-sevenths", action="store_true", help="Disable seventh chords (only triads)")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    y, sr = read_audio_mono(args.path, normalize=(not args.no_normalize))
    if args.loop_bars < 1:
        args.loop_bars = 1
    if args.bars_to_scan < 1:
        args.bars_to_scan = 1
    if args.beats_per_bar < 1:
        args.beats_per_bar = 1

    duration = len(y) / sr
    rms = rms_energy(y)
    sc = spectral_centroid_avg(y, sr)
    notes = []

    if args.force_bpm is not None and args.force_bpm > 0:
        bpm = float(args.force_bpm)
        conf = 1.0
        notes.append("Tempo forced by user.")
    else:
        bpm, conf = estimate_tempo_hybrid(y, sr)
        if bpm is None:
            notes.append("Tempo detection unstable; consider --force-bpm.")
        else:
            notes.append(f"Tempo detected ≈ {bpm:.1f} BPM (confidence {conf:.2f}).")

    key, mode = None, None
    try:
        if HAVE_LIBROSA:
            key, mode = estimate_key_librosa(y, sr)
        else:
            key, mode = estimate_key_fallback(y, sr)
        if key and mode:
            notes.append(f"Key guess: {key} {mode}.")
        else:
            notes.append("Key estimation inconclusive.")
    except Exception as e:
        notes.append(f"Key estimation skipped ({type(e).__name__}: {e}).")

    # Chords (optional)
    chord_segments = None
    if not args.no_chords:
        if HAVE_LIBROSA:
            key_hint = (key, mode) if key and mode and not args.no_chord_key_bias else None
            chord_segments = estimate_chords_librosa(
                y, sr,
                confidence_threshold=max(0.0, min(1.0, args.chord_confidence)),
                include_sevenths=(not args.no_chord_sevenths),
                key_hint=key_hint,
            )
            if chord_segments:
                notes.append(f"Detected {len(chord_segments)} chord segments.")
            else:
                notes.append("Chord detection found no stable segments.")
        else:
            notes.append("Chord detection requires librosa; skipping.")

    # Optional exports
    if args.export_click and bpm:
        click_sr = args.click_sr or sr
        # synth at click_sr for highest fidelity, duration = full file
        write_click_wav(args.export_click, click_sr, duration, bpm, strong_every=args.beats_per_bar)
        notes.append(f"Click written: {args.export_click}")

    if args.export_loop and bpm:
        start, end = extract_best_loop(
            y, sr, bpm, bars=args.loop_bars, bars_to_scan=args.bars_to_scan, beats_per_bar=args.beats_per_bar
        )
        if start is not None:
            seg = y[start:end].copy()
            # short crossfades at head/tail
            fade = int(0.01 * sr)
            if fade > 0 and seg.size >= 2 * fade:
                seg[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            os.makedirs(os.path.dirname(os.path.abspath(args.export_loop)) or ".", exist_ok=True)
            sf.write(args.export_loop, seg.astype(np.float32), sr)
            notes.append(f"Loop ({args.loop_bars} bars @ {args.beats_per_bar}/4) written: {args.export_loop}")
        else:
            notes.append("Loop extraction could not find a smooth beat-aligned segment.")

    result = AnalysisResult(
        path=os.path.abspath(args.path),
        duration_sec=float(duration),
        sample_rate=int(sr),
        rms=float(rms),
        spectral_centroid_hz_avg=float(sc),
        tempo_bpm=float(bpm) if bpm is not None else None,
        tempo_confidence=float(conf) if conf is not None else None,
        key_guess=key,
        mode_guess=mode,
        notes=" ".join(notes),
        chord_segments=chord_segments,
    )

    payload = json.dumps(asdict(result), indent=2)
    print(payload)
    if args.export_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_json)) or ".", exist_ok=True)
        with open(args.export_json, "w", encoding="utf-8") as f:
            f.write(payload)

    # Chords export
    if chord_segments:
        if args.export_chords_json:
            write_chords_json(args.export_chords_json, chord_segments)
        if args.export_chords_txt:
            write_chords_txt(args.export_chords_txt, chord_segments)
        if args.export_chords_chordpro:
            write_chords_chordpro(args.export_chords_chordpro, chord_segments)

if __name__ == "__main__":
    main()
