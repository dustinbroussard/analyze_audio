# Analyze Audio — Groove Analyzer

A simple audio analysis toolkit and GUI for estimating tempo, key, and crafting beat‑aligned loops.

## Features

- Modernized GUI with a clean, minimal aesthetic
- Optional ttkbootstrap theme for a polished, rounded look
- Tooltips, improved spacing, readable output panel
- Light/Dark theme toggle and About dialog with fade/scale
- Card components with subtle shadow for depth
- Analyze tempo, key, RMS, spectral centroid
- Practical chord detection (major/minor) with segment timings
- Adds 7th chords (maj7, min7, dom7) and key-aware smoothing
- Optional exports: metronome click and seamless loops

## Setup

1. Create a virtual environment (recommended) and install dependencies:

   - Core: `numpy`, `scipy`, `soundfile`
   - Chords/Key (recommended): `librosa`
   - UI polish (optional but recommended): `ttkbootstrap`

2. Install requirements:

   `pip install -r requirements.txt`

If `ttkbootstrap` is not installed, the GUI falls back gracefully to a refined default ttk theme.

## Usage

- CLI/Library functions live in `groove_analyzer.py`.
- GUI entry point:

  `python3 groove_gui.py`

Use the GUI to pick an audio file, optionally set export paths for a click track and loop, and run analysis.

Chord controls in the GUI:
- Toggle chord detection on/off
- Minimum chord confidence slider
- Optional export paths for JSON, TXT, and ChordPro

## Notes

- The GUI applies subtle animations (window fade-in) and improved focus/hover styling where supported.
- Native file dialogs are used for reliability; visuals follow the OS theme.
- Chord detection requires `librosa` and computes chroma + template matching for major/minor triads, returning labeled segments.
- Chord detection requires `librosa` and computes chroma + template matching for major/minor triads and 7ths, with key-aware smoothing.

### CLI chord options

- `--no-chords` disable chord detection
- `--chord-confidence 0.35` set min segment confidence
- `--no-chord-key-bias` disable key-aware scoring bias
- `--no-chord-sevenths` detect only triads
- Exports: `--export-chords-json`, `--export-chords-txt`, `--export-chords-chordpro`
