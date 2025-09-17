#!/usr/bin/env python3
import json
import os
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# Optional modern theming (rounded corners, modern palettes)
try:
    import ttkbootstrap as tb
    HAVE_TTKBOOTSTRAP = True
except Exception:
    tb = None
    HAVE_TTKBOOTSTRAP = False

# Local analyzer import (same folder)
import groove_analyzer as ga

APP_TITLE = "Groove Analyzer — Lite GUI"

# Simple tooltip helper for accessibility
class Tooltip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self._id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _):
        self._cancel()
        self._id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._id is not None:
            self.widget.after_cancel(self._id)
            self._id = None

    def _show(self):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") or (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 12
        y = y + cy + self.widget.winfo_rooty() + 12
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            background="#111827",
            foreground="#F9FAFB",
            padx=8,
            pady=4,
            font=("Segoe UI", 9),
        )
        label.pack(ipadx=2)

    def _hide(self, _):
        self._cancel()
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

class GrooveGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x680")
        self.minsize(880, 600)

        # STATE
        self.audio_path = tk.StringVar()
        self.export_click_path = tk.StringVar()
        self.export_loop_path = tk.StringVar()
        self.loop_bars = tk.IntVar(value=4)
        self.bars_to_scan = tk.IntVar(value=32)
        self.beats_per_bar = tk.IntVar(value=4)
        self.force_bpm = tk.StringVar(value="")     # string so empty is allowed
        self.no_normalize = tk.BooleanVar(value=False)
        self.click_sr = tk.StringVar(value="")      # empty = use source sr
        self.status = tk.StringVar(value="Ready.")
        self.analyzing = False
        self.theme_mode = tk.StringVar(value="dark")
        self.detect_chords = tk.BooleanVar(value=True)
        self.chord_confidence = tk.DoubleVar(value=0.35)
        self.export_chords_json = tk.StringVar()
        self.export_chords_txt = tk.StringVar()
        self.export_chords_cp = tk.StringVar()

        # Styling
        self._init_style()
        self._build_ui()
        self._fade_in()
        self._last_chords = []
        self._last_duration = 0.0
        self.bind("<Configure>", self._on_resize)

    def _init_style(self):
        try:
            if HAVE_TTKBOOTSTRAP:
                # Modern theme (rounded corners, refined palettes)
                self.style = tb.Style(theme="flatly")
                self.ttk = tb.ttk
            else:
                self.style = ttk.Style()
                # Prefer clam for better ttk styling controls
                if "clam" in self.style.theme_names():
                    self.style.theme_use("clam")
                self.ttk = ttk
            # Global fonts
            default_font = ("Segoe UI", 10)
            self.option_add("*Font", default_font)
            self.option_add("*TButton.padding", 8)

            # Apply initial palette and styles
            self._apply_theme(self.theme_mode.get())
        except Exception:
            # Fail safely
            self.ttk = ttk

    def _apply_theme(self, mode: str):
        # Palette presets
        if mode == "light":
            self.bg = "#F3F4F6"
            self.surface = "#FFFFFF"
            self.card = "#FFFFFF"
            self.shadow = "#E5E7EB"
            self.border = "#D1D5DB"
            self.text_primary = "#111827"
            self.text_secondary = "#4B5563"
            self.accent = "#2563EB"
        else:
            self.bg = "#0B1220"
            self.surface = "#0F172A"
            self.card = "#111827"
            self.shadow = "#0A1225"
            self.border = "#1F2937"
            self.text_primary = "#E5E7EB"
            self.text_secondary = "#9CA3AF"
            self.accent = "#3B82F6"

        self.configure(bg=self.bg)
        self._refresh_styles()
        # Live widgets that need manual recolor
        if hasattr(self, "output") and isinstance(self.output, tk.Text):
            self.output.configure(bg=self.bg if self.bg != self.card else self.surface, fg=self.text_primary, insertbackground=self.text_primary)

    def _refresh_styles(self):
        # Cards and labels
        self.style.configure("Card.TFrame", background=self.card, borderwidth=1, relief="flat")
        self.style.configure("Section.TLabelframe", background=self.card, foreground=self.text_primary, borderwidth=1)
        self.style.configure("Section.TLabelframe.Label", background=self.card, foreground=self.text_secondary)
        self.style.configure("TLabel", background=self.card, foreground=self.text_primary)
        self.style.configure("Muted.TLabel", background=self.card, foreground=self.text_secondary)
        # Buttons
        if not HAVE_TTKBOOTSTRAP:
            self.style.configure("Primary.TButton", padding=10)
            self.style.map(
                "Primary.TButton",
                background=[("active", self.accent)],
                relief=[("pressed", "sunken"), ("!pressed", "raised")],
            )
        # Inputs
        entry_bg = "#F9FAFB" if self.theme_mode.get() == "light" else "#0B1220"
        focus_bg = "#FFFFFF" if self.theme_mode.get() == "light" else "#0A1225"
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=self.text_primary)
        self.style.map("TEntry", fieldbackground=[("focus", focus_bg)])
        self.style.configure("TSpinbox", fieldbackground=entry_bg, foreground=self.text_primary)
        self.style.configure("TCheckbutton", background=self.card, foreground=self.text_primary)
        # Progressbar
        trough = "#E5E7EB" if self.theme_mode.get() == "light" else "#0B1220"
        self.style.configure("TProgressbar", troughcolor=trough, background=self.accent)

    def _make_card(self, parent, title=None, padding=14, fill="x", expand=False):
        # Outer shadow container
        shadow = tk.Frame(parent, bg=self.shadow, highlightthickness=0, bd=0)
        shadow.pack(fill=fill, expand=expand, padx=12, pady=(4, 8))
        if title:
            card = ttk.LabelFrame(shadow, text=title, padding=padding, style="Section.TLabelframe")
        else:
            card = ttk.Frame(shadow, padding=padding, style="Card.TFrame")
        card.pack(fill=fill, expand=expand)
        return card

    def _fade_in(self, duration_ms: int = 220):
        try:
            self.attributes("-alpha", 0.0)
            steps = 12
            step = 1 / steps
            delay = max(10, duration_ms // steps)

            def _step(a=0.0):
                a += step
                self.attributes("-alpha", min(1.0, a))
                if a < 1.0:
                    self.after(delay, _step, a)

            self.after(20, _step)
        except Exception:
            pass

    def _build_ui(self):
        # === HEADER ===
        header = self._make_card(self, padding=14)
        ttk.Label(header, text="Groove Analyzer", style="TLabel", font=("Segoe UI Semibold", 14)).pack(side="left")
        ttk.Label(header, text="Analyze tempo, key, and groove quickly", style="Muted.TLabel").pack(side="left", padx=(10,0))
        # Right side controls: theme toggle + about
        right = ttk.Frame(header, style="Card.TFrame")
        right.pack(side="right")
        self.theme_btn = ttk.Checkbutton(
            right,
            text="Dark",
            variable=self.theme_mode,
            onvalue="dark",
            offvalue="light",
            command=self._toggle_theme,
            style="TCheckbutton",
        )
        self.theme_btn.pack(side="right", padx=(0,8))
        Tooltip(self.theme_btn, "Toggle light/dark theme")
        about_btn = ttk.Button(right, text="About", command=self._open_about)
        about_btn.pack(side="right", padx=(0,8))
        Tooltip(about_btn, "About Groove Analyzer")

        # === TOP: file pickers & options ===
        top = self._make_card(self, padding=14)

        # Audio file
        l_audio = ttk.Label(top, text="Audio file:")
        l_audio.grid(row=0, column=0, sticky="w", pady=2)
        e_audio = ttk.Entry(top, textvariable=self.audio_path, width=80)
        e_audio.grid(row=0, column=1, padx=6, sticky="we")
        b_audio = ttk.Button(top, text="Browse…", command=self._choose_audio, style=("TButton" if HAVE_TTKBOOTSTRAP else "Primary.TButton"))
        b_audio.grid(row=0, column=2)
        Tooltip(b_audio, "Choose a local audio file to analyze")

        # Export click
        ttk.Label(top, text="Click WAV (optional):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(top, textvariable=self.export_click_path, width=80).grid(row=1, column=1, padx=6, sticky="we")
        b_click = ttk.Button(top, text="Save as…", command=self._choose_click)
        b_click.grid(row=1, column=2)
        Tooltip(b_click, "Save a metronome click aligned to the detected tempo")

        # Export loop
        ttk.Label(top, text="Loop WAV (optional):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(top, textvariable=self.export_loop_path, width=80).grid(row=2, column=1, padx=6, sticky="we")
        b_loop = ttk.Button(top, text="Save as…", command=self._choose_loop)
        b_loop.grid(row=2, column=2)
        Tooltip(b_loop, "Export a beat-aligned seamless audio loop")

        # Grid weights
        top.columnconfigure(1, weight=1)

        # === OPTIONS ===
        opts = self._make_card(self, title="Options", padding=12)

        # Row 0: loop bars, bars_to_scan, beats_per_bar
        ttk.Label(opts, text="Loop bars").grid(row=0, column=0, sticky="w", pady=3)
        sp_loop = ttk.Spinbox(opts, from_=1, to=64, textvariable=self.loop_bars, width=6)
        sp_loop.grid(row=0, column=1, padx=(6,20))
        Tooltip(sp_loop, "Number of bars to include in exported loop")

        ttk.Label(opts, text="Bars to scan").grid(row=0, column=2, sticky="w", pady=3)
        sp_scan = ttk.Spinbox(opts, from_=1, to=128, textvariable=self.bars_to_scan, width=6)
        sp_scan.grid(row=0, column=3, padx=(6,20))
        Tooltip(sp_scan, "How many bars to search for the cleanest loop")

        ttk.Label(opts, text="Beats / bar").grid(row=0, column=4, sticky="w", pady=3)
        sp_bpb = ttk.Spinbox(opts, from_=1, to=12, textvariable=self.beats_per_bar, width=6)
        sp_bpb.grid(row=0, column=5, padx=(6,20))
        Tooltip(sp_bpb, "Time signature numerator (beats per bar)")

        # Row 1: force bpm, click sr, normalize
        ttk.Label(opts, text="Force BPM (optional)").grid(row=1, column=0, sticky="w", pady=3)
        e_bpm = ttk.Entry(opts, textvariable=self.force_bpm, width=10)
        e_bpm.grid(row=1, column=1, padx=(6,20), sticky="w")
        Tooltip(e_bpm, "Override tempo detection with a fixed BPM")

        ttk.Label(opts, text="Click SR (Hz, optional)").grid(row=1, column=2, sticky="w", pady=3)
        e_sr = ttk.Entry(opts, textvariable=self.click_sr, width=10)
        e_sr.grid(row=1, column=3, padx=(6,20), sticky="w")
        Tooltip(e_sr, "Sample rate for exported click track (blank = source)")

        cb_norm = ttk.Checkbutton(opts, text="Disable input normalization", variable=self.no_normalize)
        cb_norm.grid(row=1, column=4, columnspan=2, sticky="w")
        Tooltip(cb_norm, "Keep original level instead of normalizing input audio")

        # Row 2: chord detection toggle + confidence
        cb_chords = ttk.Checkbutton(opts, text="Detect chords (librosa)", variable=self.detect_chords)
        cb_chords.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,0))
        Tooltip(cb_chords, "Estimate chord segments using chroma templates. Disable to save CPU.")
        ttk.Label(opts, text="Min chord confidence").grid(row=2, column=2, sticky="e", padx=(12,4))
        sc = ttk.Scale(opts, from_=0.0, to=1.0, orient="horizontal", variable=self.chord_confidence)
        sc.grid(row=2, column=3, sticky="we")
        Tooltip(sc, "Segments below this confidence are dropped")
        opts.columnconfigure(3, weight=1)

        # Row 3-5: chord export paths
        ttk.Label(opts, text="Chords JSON (optional)").grid(row=3, column=0, sticky="w", pady=(8,2))
        ttk.Entry(opts, textvariable=self.export_chords_json, width=60).grid(row=3, column=1, columnspan=3, padx=6, sticky="we")
        b_cjson = ttk.Button(opts, text="Save as…", command=self._choose_chords_json)
        b_cjson.grid(row=3, column=4)
        Tooltip(b_cjson, "Save chord segments as JSON")

        ttk.Label(opts, text="Chords TXT (optional)").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(opts, textvariable=self.export_chords_txt, width=60).grid(row=4, column=1, columnspan=3, padx=6, sticky="we")
        b_ctxt = ttk.Button(opts, text="Save as…", command=self._choose_chords_txt)
        b_ctxt.grid(row=4, column=4)
        Tooltip(b_ctxt, "Save chord segments as plain text")

        ttk.Label(opts, text="ChordPro (optional)").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Entry(opts, textvariable=self.export_chords_cp, width=60).grid(row=5, column=1, columnspan=3, padx=6, sticky="we")
        b_ccp = ttk.Button(opts, text="Save as…", command=self._choose_chords_cp)
        b_ccp.grid(row=5, column=4)
        Tooltip(b_ccp, "Save chord segments in ChordPro style")

        # Buttons
        buttons = ttk.Frame(self, padding=(12, 8), style="Card.TFrame")
        buttons.pack(fill="x", padx=12)

        self.run_btn = ttk.Button(
            buttons,
            text="Analyze",
            command=self._run_analysis,
            style=("TButton" if HAVE_TTKBOOTSTRAP else "Primary.TButton"),
        )
        self.run_btn.pack(side="left")

        ttk.Button(
            buttons,
            text="Analyze + Export",
            command=self._run_analysis_export,
        ).pack(side="left", padx=8)

        # Status & progress
        status_bar = self._make_card(self, padding=8)
        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=160)
        self.progress.pack(side="left", padx=(0,10))
        ttk.Label(status_bar, textvariable=self.status, style="Muted.TLabel").pack(side="left")

        # === OUTPUT ===
        outbox = self._make_card(self, title="Output", padding=8, fill="both", expand=True)

        # Readable monospaced output with subtle elevation
        self.output = tk.Text(outbox, wrap="word", height=18, bg=self.bg if self.bg != self.card else self.surface, fg=self.text_primary, insertbackground=self.text_primary, relief="flat")
        self.output.pack(fill="both", expand=True, padx=4, pady=4)
        self._write("Ready.\n")

        # === CHORDS TIMELINE ===
        chords_box = self._make_card(self, title="Chords", padding=8, fill="x", expand=False)
        self.chords_canvas = tk.Canvas(chords_box, height=120, highlightthickness=0, bd=0, bg=self.card)
        self.chords_canvas.pack(fill="x", expand=False)
        self.chords_empty = ttk.Label(chords_box, text="Run analysis to view chord timeline", style="Muted.TLabel")
        self.chords_empty.pack(anchor="w", padx=4, pady=6)

    # ---------- UI helpers ----------
    def _choose_audio(self):
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a *.aiff *.aif"), ("All files", "*.*")]
        )
        if path:
            self.audio_path.set(path)

    def _choose_click(self):
        path = filedialog.asksaveasfilename(
            title="Save click WAV",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")]
        )
        if path:
            self.export_click_path.set(path)

    def _choose_loop(self):
        path = filedialog.asksaveasfilename(
            title="Save loop WAV",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")]
        )
        if path:
            self.export_loop_path.set(path)

    def _choose_chords_json(self):
        path = filedialog.asksaveasfilename(
            title="Save chords JSON",
            defaultextension=".chords.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.export_chords_json.set(path)

    def _choose_chords_txt(self):
        path = filedialog.asksaveasfilename(
            title="Save chords TXT",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")]
        )
        if path:
            self.export_chords_txt.set(path)

    def _choose_chords_cp(self):
        path = filedialog.asksaveasfilename(
            title="Save chords ChordPro",
            defaultextension=".pro",
            filetypes=[("ChordPro", "*.pro *.cho"), ("All files", "*.*")]
        )
        if path:
            self.export_chords_cp.set(path)

    def _write(self, text: str):
        self.output.insert("end", text)
        self.output.see("end")

    def _lock_ui(self, locked: bool):
        self.analyzing = locked
        state = "disabled" if locked else "normal"
        self.run_btn.configure(state=state)
        # Start/stop spinner
        if locked:
            self.progress.start(80)
        else:
            self.progress.stop()

    def _toggle_theme(self):
        mode = self.theme_mode.get()
        self._apply_theme(mode)
        try:
            self.theme_btn.configure(text="Dark" if mode == "dark" else "Light")
        except Exception:
            pass

    def _open_about(self):
        if getattr(self, "_about_open", False):
            return
        self._about_open = True
        top = tk.Toplevel(self)
        top.withdraw()
        top.transient(self)
        top.grab_set()
        top.title("About Groove Analyzer")
        try:
            top.attributes("-alpha", 0.0)
        except Exception:
            pass
        top.configure(bg=self.shadow)

        # Card inside for content
        container = tk.Frame(top, bg=self.shadow)
        container.pack(fill="both", expand=True, padx=12, pady=12)
        card = ttk.Frame(container, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True)

        ttk.Label(card, text="Groove Analyzer", font=("Segoe UI Semibold", 13)).pack(anchor="w")
        ttk.Label(card, text="Analyze tempo, key, RMS, and spectral features.\nCreate metronome clicks and seamless loops.", style="Muted.TLabel").pack(anchor="w", pady=(6, 10))
        ttk.Label(card, text="© 2025", style="Muted.TLabel").pack(anchor="w")

        btns = ttk.Frame(card, style="Card.TFrame")
        btns.pack(fill="x", pady=(12,0))
        ttk.Button(btns, text="OK", command=lambda: self._close_about(top)).pack(side="right")

        # Center and animate fade+scale
        top.update_idletasks()
        w, h = max(360, top.winfo_reqwidth()), max(180, top.winfo_reqheight())
        W, H = self.winfo_width(), self.winfo_height()
        X, Y = self.winfo_rootx(), self.winfo_rooty()
        cx, cy = X + W // 2, Y + H // 2
        scale_steps = 10
        start_scale = 0.9
        for i in range(scale_steps + 1):
            f = start_scale + (1 - start_scale) * (i / scale_steps)
            cw, ch = int(w * f), int(h * f)
            x = cx - cw // 2
            y = cy - ch // 2
            top.geometry(f"{cw}x{ch}+{x}+{y}")
            top.deiconify()
            try:
                top.attributes("-alpha", i / scale_steps)
            except Exception:
                pass
            top.update_idletasks()
            top.after(12)

        top.bind("<Escape>", lambda e: self._close_about(top))

    def _close_about(self, top: tk.Toplevel):
        # Fade out
        try:
            for i in range(10, -1, -1):
                top.attributes("-alpha", i / 10.0)
                top.update_idletasks()
                top.after(12)
        except Exception:
            pass
        try:
            top.grab_release()
        except Exception:
            pass
        top.destroy()
        self._about_open = False

    # ---------- Analysis runners ----------
    def _collect_params(self):
        # Validate/cast entries
        try:
            force_bpm_val = float(self.force_bpm.get()) if self.force_bpm.get().strip() else None
        except ValueError:
            force_bpm_val = None
        try:
            click_sr_val = int(self.click_sr.get()) if self.click_sr.get().strip() else None
        except ValueError:
            click_sr_val = None

        return dict(
            audio_path=self.audio_path.get().strip(),
            export_click=self.export_click_path.get().strip() or None,
            export_loop=self.export_loop_path.get().strip() or None,
            loop_bars=max(1, int(self.loop_bars.get())),
            bars_to_scan=max(1, int(self.bars_to_scan.get())),
            beats_per_bar=max(1, int(self.beats_per_bar.get())),
            force_bpm=force_bpm_val,
            no_normalize=bool(self.no_normalize.get()),
            click_sr=click_sr_val,
            detect_chords=bool(self.detect_chords.get()),
            chord_confidence=float(self.chord_confidence.get()),
            export_chords_json=self.export_chords_json.get().strip() or None,
            export_chords_txt=self.export_chords_txt.get().strip() or None,
            export_chords_cp=self.export_chords_cp.get().strip() or None,
        )

    def _run_analysis(self):
        self._start_job(export=False)

    def _run_analysis_export(self):
        self._start_job(export=True)

    def _start_job(self, export: bool):
        if self.analyzing:
            return
        params = self._collect_params()
        if not params["audio_path"]:
            messagebox.showwarning("Missing file", "Choose an audio file first.")
            return
        self.status.set("Working…")
        self._lock_ui(True)
        self.output.delete("1.0", "end")
        self._write("Running analysis…\n")

        t = threading.Thread(target=self._job, args=(params, export), daemon=True)
        t.start()

    def _job(self, params, export):
        try:
            result = self._analyze_core(params, export)
            # Pretty print
            payload = json.dumps(result, indent=2)
            self.after(0, self._on_success, payload)
        except Exception as e:
            tb = traceback.format_exc()
            self.after(0, self._on_error, str(e), tb)

    def _on_success(self, payload: str):
        self._write(payload + "\n")
        self.status.set("Done.")
        self._lock_ui(False)
        # Render chord timeline
        try:
            data = json.loads(payload)
            segs = data.get("chord_segments") or []
            duration = float(data.get("duration_sec") or 0.0)
            self._render_chord_timeline(segs, duration)
            self._last_chords = segs
            self._last_duration = duration
        except Exception:
            pass

    def _on_error(self, msg: str, tb: str):
        self._write(f"[ERROR] {msg}\n{tb}\n")
        self.status.set("Error.")
        self._lock_ui(False)

    # ---------- Core orchestration ----------
    def _analyze_core(self, p, export):
        path = p["audio_path"]
        y, sr = ga.read_audio_mono(path, normalize=(not p["no_normalize"]))
        duration = len(y) / sr

        rms = ga.rms_energy(y)
        sc = ga.spectral_centroid_avg(y, sr)
        notes = []
        chord_segments = None

        # Tempo
        if p["force_bpm"] and p["force_bpm"] > 0:
            bpm = float(p["force_bpm"])
            conf = 1.0
            notes.append("Tempo forced by user.")
        else:
            bpm, conf = ga.estimate_tempo_hybrid(y, sr)
            if bpm is None:
                notes.append("Tempo detection unstable; consider forcing BPM.")
            else:
                notes.append(f"Tempo detected ≈ {bpm:.1f} BPM (confidence {conf:.2f}).")

        # Key (librosa → fallback)
        key, mode = None, None
        try:
            if ga.HAVE_LIBROSA:
                key, mode = ga.estimate_key_librosa(y, sr)
            else:
                key, mode = ga.estimate_key_fallback(y, sr)
            if key and mode:
                notes.append(f"Key guess: {key} {mode}.")
            else:
                notes.append("Key estimation inconclusive.")
        except Exception as e:
            notes.append(f"Key estimation skipped ({type(e).__name__}: {e}).")

        # Chords (optional)
        if p.get("detect_chords"):
            try:
                if ga.HAVE_LIBROSA:
                    # provide key hint to bias if available later
                    chord_segments = ga.estimate_chords_librosa(
                        y, sr,
                        confidence_threshold=float(p.get("chord_confidence") or 0.3),
                        include_sevenths=True,
                        key_hint=(key, mode),
                    )
                    if chord_segments:
                        notes.append(f"Detected {len(chord_segments)} chord segments.")
                    else:
                        notes.append("Chord detection found no stable segments.")
                else:
                    notes.append("Chord detection requires librosa; skipping.")
            except Exception as e:
                notes.append(f"Chord detection failed ({type(e).__name__}: {e}).")

        # Optional exports
        if export and bpm:
            # click
            if p["export_click"]:
                click_sr = p["click_sr"] or sr
                ga.write_click_wav(p["export_click"], click_sr, duration, bpm, strong_every=p["beats_per_bar"])
                notes.append(f"Click written: {p['export_click']}")
            # loop
            if p["export_loop"]:
                start, end = ga.extract_best_loop(
                    y, sr, bpm,
                    bars=p["loop_bars"], bars_to_scan=p["bars_to_scan"], beats_per_bar=p["beats_per_bar"]
                )
                if start is not None:
                    seg = y[start:end].copy()
                    fade = int(0.01 * sr)
                    if fade > 0 and seg.size >= 2 * fade:
                        seg[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                        seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
                    os.makedirs(os.path.dirname(os.path.abspath(p["export_loop"])) or ".", exist_ok=True)
                    import soundfile as sf
                    sf.write(p["export_loop"], seg.astype(np.float32), sr)
                    notes.append(f"Loop ({p['loop_bars']} bars @ {p['beats_per_bar']}/4) written: {p['export_loop']}")
                else:
                    notes.append("Loop extraction could not find a smooth beat-aligned segment.")

        # Chord exports (if any paths set)
        if export and chord_segments:
            try:
                if p.get("export_chords_json"):
                    ga.write_chords_json(p["export_chords_json"], chord_segments)
                    notes.append(f"Chords JSON written: {p['export_chords_json']}")
                if p.get("export_chords_txt"):
                    ga.write_chords_txt(p["export_chords_txt"], chord_segments)
                    notes.append(f"Chords TXT written: {p['export_chords_txt']}")
                if p.get("export_chords_cp"):
                    ga.write_chords_chordpro(p["export_chords_cp"], chord_segments)
                    notes.append(f"ChordPro written: {p['export_chords_cp']}")
            except Exception as e:
                notes.append(f"Chord export failed ({type(e).__name__}: {e}).")

        result = ga.AnalysisResult(
            path=os.path.abspath(path),
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
        return result.__dict__

    def _render_chord_timeline(self, segments: list, duration: float):
        canvas = self.chords_canvas
        canvas.delete("all")
        if not segments or duration <= 0:
            try:
                self.chords_empty.configure(text="No chords detected")
                self.chords_empty.lift()
            except Exception:
                pass
            return
        else:
            try:
                self.chords_empty.configure(text="")
            except Exception:
                pass
        w = canvas.winfo_width() or canvas.winfo_reqwidth()
        h = canvas.winfo_height() or 120
        pad = 6
        y0, y1 = pad, h - pad

        def color_for_label(label: str) -> str:
            # map root to hue
            root = (label.split(":", 1)[0]).strip()
            try:
                idx = ga.KEY_NAMES.index(root)
            except Exception:
                idx = 0
            hue = (idx / 12.0)
            # simple HSL to RGB
            import colorsys
            light = 0.32 if self.theme_mode.get() == "dark" else 0.78
            sat = 0.55 if self.theme_mode.get() == "dark" else 0.45
            r, g, b = colorsys.hls_to_rgb(hue, light, sat)
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        for seg in segments:
            s = float(seg.get("start_sec", 0.0))
            e = float(seg.get("end_sec", s))
            if e <= s:
                continue
            x0 = pad + (s / duration) * (w - 2 * pad)
            x1 = pad + (e / duration) * (w - 2 * pad)
            fill = color_for_label(seg.get("label", "C:maj"))
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=self.border)
            # label if wide enough
            if (x1 - x0) > 50:
                canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=seg.get("label", ""), fill=self.text_primary, font=("Segoe UI", 10))

    def _on_resize(self, event):
        if getattr(self, "_last_chords", None) and self._last_duration:
            # debounce via after_idle
            try:
                if getattr(self, "_resize_pending", False):
                    return
                self._resize_pending = True
                def _run():
                    self._render_chord_timeline(self._last_chords, self._last_duration)
                    self._resize_pending = False
                self.after_idle(_run)
            except Exception:
                pass

def main():
    app = GrooveGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
