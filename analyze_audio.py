#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf
from scipy import signal

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
    tempo_bpm: float | None
    tempo_confidence: float | None
    key_guess: str | None
    mode_guess: str | None
    notes: str

KEY_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def read_audio_mono(path: str):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    # normalize lightly to avoid weird RMS scaling differences
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32), sr

def estimate_tempo_hybrid(y: np.ndarray, sr: int):
    """
    Hybrid tempo estimate that doesn't need librosa:
    1) Energy-based onset curve
    2) Autocorrelation to get periodicity
    Returns (bpm, confidence)
    """
    # Downsample for speed
    target_sr = 11025
    if sr > target_sr:
        decim = sr // target_sr
        y_ds = signal.decimate(y, decim, zero_phase=True)
        sr_ds = sr // decim
    else:
        y_ds, sr_ds = y, sr

    # Onset proxy: rectified high-pass energy
    b = signal.firwin(101, cutoff=60, fs=sr_ds, pass_zero=False)
    y_hp = signal.lfilter(b, [1.0], y_ds)
    onset_env = np.abs(y_hp)
    onset_env = signal.medfilt(onset_env, kernel_size=5)
    onset_env = onset_env - np.median(onset_env)
    onset_env[onset_env < 0] = 0

    # Frame the onset envelope to ~ 200 Hz for stability
    hop = max(1, int(sr_ds / 200))
    framed = onset_env[::hop]
    framed -= framed.mean()
    if np.std(framed) > 1e-6:
        framed /= np.std(framed)

    # Autocorrelation
    corr = signal.correlate(framed, framed, mode='full')
    corr = corr[len(corr)//2:]

    # Ignore very short lags and very long (BPM bounds)
    fps = sr_ds / hop
    min_bpm, max_bpm = 40, 200  # sensible bounds
    min_lag = int(fps * 60 / max_bpm)
    max_lag = int(fps * 60 / min_bpm)
    if max_lag > len(corr)-1:
        max_lag = len(corr)-1
    if max_lag <= min_lag+5:
        return None, None

    roi = corr[min_lag:max_lag]
    if len(roi) < 3 or np.max(roi) <= 0:
        return None, None

    peaks, _ = signal.find_peaks(roi, distance=int(fps*60/max_bpm))
    if len(peaks) == 0:
        return None, None

    # Best peak
    best_idx = peaks[np.argmax(roi[peaks])]
    best_lag = min_lag + best_idx
    bpm = 60.0 * fps / best_lag

    # Confidence: peak sharpness relative to neighborhood
    neighborhood = roi[max(0, best_idx-10):best_idx+11]
    conf = float(roi[best_idx] / (np.mean(neighborhood) + 1e-6))
    # Fit BPM into sane drum’n’groove range by octave folding
    while bpm < 70:
        bpm *= 2
    while bpm > 170:
        bpm /= 2

    return float(bpm), conf

def spectral_centroid_avg(y: np.ndarray, sr: int):
    freqs, times, Sxx = signal.spectrogram(y, sr, nperseg=2048, noverlap=1536)
    # Avoid divide-by-zero
    S = Sxx + 1e-12
    centroid = np.sum(freqs[:, None] * S, axis=0) / np.sum(S, axis=0)
    return float(np.mean(centroid))

def rms_energy(y: np.ndarray):
    return float(np.sqrt(np.mean(y**2)))

def estimate_key_librosa(y: np.ndarray, sr: int):
    """Rough key/mode via chroma + Krumhansl profiles (requires librosa)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_norm = chroma / (np.max(chroma, axis=0, keepdims=True) + 1e-9)
    chroma_mean = np.mean(chroma_norm, axis=1)

    # Krumhansl key profiles (major/minor)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    scores = []
    for i in range(12):
        maj_score = np.dot(np.roll(major_profile, i), chroma_mean)
        min_score = np.dot(np.roll(minor_profile, i), chroma_mean)
        scores.append((i, 'major', maj_score))
        scores.append((i, 'minor', min_score))
    best = max(scores, key=lambda x: x[2])
    key_name = KEY_NAMES[best[0]]
    return key_name, best[1]

def write_click_wav(out_path: str, sr: int, duration_sec: float, bpm: float, strong_every=4):
    """Simple metronome click (strong beat accent)."""
    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    t = np.linspace(0, duration_sec, int(sr*duration_sec), endpoint=False)
    audio = np.zeros_like(t)
    spb = 60.0 / bpm
    n_beats = int(duration_sec / spb)
    click_len = int(0.02 * sr)  # 20 ms
    for b in range(n_beats):
        start = int(b * spb * sr)
        end = min(start + click_len, len(audio))
        # Strong beat every 'strong_every' beats
        freq = 2000.0 if (b % strong_every == 0) else 1000.0
        click = 0.8 * np.hanning(end - start) * np.sin(2*np.pi*freq*np.arange(end-start)/sr)
        audio[start:end] += click
    # Normalize
    m = np.max(np.abs(audio))
    if m > 0:
        audio = audio / m * 0.9
    sf.write(out_path, audio.astype(np.float32), sr)

def extract_best_loop(y: np.ndarray, sr: int, bpm: float, bars: int = 4, bars_to_scan: int = 32):
    """
    Naive 'best loop' finder:
    - assume 4/4, compute beats from bpm
    - scan the first ~bars_to_scan bars for a segment whose start/end RMS match
    - return (start_sample, end_sample)
    """
    if bpm is None or bpm <= 0:
        return None, None

    spb = 60.0 / bpm
    spp = spb * 4          # seconds per bar at 4/4
    seg_len = int(bars * spp * sr)
    total_bars = int((len(y) / sr) / spp)
    scan_bars = min(total_bars - bars, bars_to_scan)
    if scan_bars <= 1:
        return None, None

    best_score, best_start = -1.0, None
    for b in range(scan_bars):
        start = int(b * spp * sr)
        end = start + seg_len
        if end >= len(y): break
        seg = y[start:end]
        # Score: similarity of first 200ms vs last 200ms + low crest factor (loop smoothness)
        win = int(0.2 * sr)
        head = seg[:win]
        tail = seg[-win:]
        if len(head) < win or len(tail) < win: continue
        sim = np.corrcoef(head, tail)[0,1]
        if np.isnan(sim): sim = -1.0
        # penalize loud transients (hard cuts)
        crest = np.max(np.abs(seg)) / (np.sqrt(np.mean(seg**2)) + 1e-9)
        score = sim - 0.1 * (crest - 1.0)
        if score > best_score:
            best_score = score
            best_start = start

    if best_start is None:
        return None, None
    return best_start, best_start + seg_len

def main():
    ap = argparse.ArgumentParser(description="CPU-friendly audio groove analyzer")
    ap.add_argument("path", help="Path to audio file (mp3/wav/flac/ogg)")
    ap.add_argument("--export-click", metavar="CLICK_WAV", help="Write a metronome click WAV")
    ap.add_argument("--export-loop", metavar="LOOP_WAV", help="Write a 4–8 bar loop WAV")
    ap.add_argument("--loop-bars", type=int, default=4, help="Bars to include in exported loop (default 4)")
    ap.add_argument("--force-bpm", type=float, default=None, help="Override/force BPM if detector fails")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    y, sr = read_audio_mono(args.path)
    if args.loop_bars < 1:
        args.loop_bars = 1
    duration = len(y) / sr
    rms = rms_energy(y)
    sc = spectral_centroid_avg(y, sr)
    notes = []

    if args.force_bpm is not None:
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
    if HAVE_LIBROSA:
        try:
            key, mode = estimate_key_librosa(y, sr)
            notes.append(f"Key guess: {key} {mode}.")
        except Exception as e:
            notes.append(f"Key estimation skipped (librosa error: {e}).")
    else:
        notes.append("Key estimation skipped (librosa not installed).")

    # Optional exports
    if args.export_click and bpm:
        write_click_wav(args.export_click, sr, duration, bpm)
        notes.append(f"Click written: {args.export_click}")

    if args.export_loop and bpm:
        start, end = extract_best_loop(y, sr, bpm, bars=args.loop_bars)
        if start is not None:
            seg = y[start:end]
            # apply short fades to avoid clicks
            fade = int(0.01 * sr)
            seg[:fade] = seg[:fade] * np.linspace(0,1,fade)
            seg[-fade:] = seg[-fade:] * np.linspace(1,0,fade)
            # Ensure output directory exists
            out_dir = os.path.dirname(os.path.abspath(args.export_loop))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            sf.write(args.export_loop, seg, sr)
            notes.append(f"Loop ({args.loop_bars} bars) written: {args.export_loop}")
        else:
            notes.append("Loop extraction could not find a smooth segment.")

    result = AnalysisResult(
        path=os.path.abspath(args.path),
        duration_sec=duration,
        sample_rate=sr,
        rms=rms,
        spectral_centroid_hz_avg=sc,
        tempo_bpm=bpm,
        tempo_confidence=conf,
        key_guess=key,
        mode_guess=mode,
        notes=" ".join(notes),
    )
    print(json.dumps(asdict(result), indent=2))

if __name__ == "__main__":
    main()
