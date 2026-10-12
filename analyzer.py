#!/usr/bin/env python3
"""
Reel Hook Analyzer v1
Honest, code-measured virality heuristics for short videos.
NO fake "brain scan" claims: everything here is computed from pixels and audio.

Usage:
    python analyzer.py video.mp4 [--json out.json]

Metrics (0-100 each, weighted into a total score):
  1. HOOK (40%)  - first 3 seconds: motion intensity, audio energy, brightness punch
  2. PACING (30%) - cuts per 10s via ffmpeg scene detection
  3. AUDIO (20%)  - energy consistency: biggest drop-off between 1s windows
  4. TEXT (10%)   - on-screen text likelihood via edge-density heuristic (crude v1)

Requires: ffmpeg + ffprobe on PATH, numpy.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# --- calibration constants (documented in README) ---
# MOTION_REF: p90 of lively code-animation ~= 7.5 (measured on real reels, 320x180 @10fps)
MOTION_REF = 8.0
PACING_MOTION_REF = 6.0   # sustained-motion component of pacing
AUDIO_REF = 0.30        # RMS that maps to 1.0
EDGE_REF = 40.0         # mean sobel magnitude that maps to 1.0
CUTS_REF = 8.0          # cuts per 10s that maps to 1.0
SCENE_THRESH = 0.4


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=kw.pop("text", True), **kw)


def probe_duration(path):
    r = run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path])
    return float(r.stdout.strip())


def grab_frames(path, fps=10, ss=0.0, dur=None, w=320, h=180):
    """Raw RGB frames via ffmpeg pipe -> numpy (N, h, w, 3)."""
    cmd = [FFMPEG, "-v", "error", "-ss", str(ss)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-i", path, "-vf", f"fps={fps},scale={w}:{h}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, capture_output=True)
    n = len(p.stdout) // (w * h * 3)
    if n == 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8)
    return np.frombuffer(p.stdout, dtype=np.uint8).reshape(n, h, w, 3)


def grab_audio(path, sr=16000):
    """Mono float32 PCM via ffmpeg pipe."""
    cmd = [FFMPEG, "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def motion_intensity(frames):
    if len(frames) < 2:
        return 0.0
    diffs = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
    return float(np.clip(diffs.mean() / MOTION_REF, 0.0, 1.0))


def brightness_punch(frames):
    if len(frames) == 0:
        return 0.0
    luma = frames.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
    return float(np.clip(luma.mean(), 0.0, 1.0))


def audio_energy(audio, sr, t0, t1):
    s0, s1 = int(t0 * sr), int(t1 * sr)
    seg = audio[s0:s1]
    if len(seg) == 0:
        return 0.0
    return float(np.clip(np.sqrt(np.mean(seg ** 2)) / AUDIO_REF, 0.0, 1.0))


def count_cuts(path, duration):
    cmd = [FFMPEG, "-v", "info", "-i", path,
           "-vf", f"select='gt(scene,{SCENE_THRESH})',showinfo",
           "-f", "null", "-"]
    r = run(cmd)
    n_cuts = sum(1 for line in r.stderr.splitlines() if "showinfo" in line and " n:" in line)
    per10 = n_cuts / max(duration, 0.1) * 10.0
    return n_cuts, float(np.clip(per10 / CUTS_REF, 0.0, 1.0)), per10


def audio_drop_penalty(audio, sr, duration):
    """Biggest relative drop between consecutive 1s RMS windows -> 0..1 penalty."""
    n = max(int(duration), 1)
    rms = []
    for i in range(n):
        s = audio[i * sr:(i + 1) * sr]
        rms.append(np.sqrt(np.mean(s ** 2)) if len(s) else 0.0)
    rms = np.array(rms)
    peak = rms.max() if rms.max() > 1e-6 else 1e-6
    drops = [(rms[i] - rms[i + 1]) / peak for i in range(len(rms) - 1)]
    max_drop = max(drops) if drops else 0.0
    return float(np.clip(1.0 - max(0.0, max_drop), 0.0, 1.0)), [float(x) for x in rms]


def text_presence(frames):
    """Crude v1 heuristic: high edge density ~= on-screen text/graphics."""
    if len(frames) == 0:
        return 0.0
    gray = frames.astype(np.float32).mean(axis=3)
    gx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    edge = (gx + gy).mean()
    return float(np.clip(edge / EDGE_REF, 0.0, 1.0))


def analyze(path):
    duration = probe_duration(path)
    sr = 16000
    audio = grab_audio(path, sr)

    hook_frames = grab_frames(path, fps=10, ss=0.0, dur=min(3.0, duration))
    all_frames = grab_frames(path, fps=3, ss=0.0, dur=duration)
    all_frames_motion = float(np.abs(all_frames[1:].astype(np.float32) - all_frames[:-1].astype(np.float32)).mean()) if len(all_frames) > 1 else 0.0

    m_hook_motion = motion_intensity(hook_frames)
    m_hook_audio = audio_energy(audio, sr, 0.0, min(3.0, duration))
    m_hook_bright = brightness_punch(hook_frames)
    hook = 0.45 * m_hook_motion + 0.35 * m_hook_audio + 0.20 * m_hook_bright

    n_cuts, m_cuts, cuts_per10 = count_cuts(path, duration)
    # Pacing blends hard cuts with sustained motion: one-shot animations
    # have no cuts, but steady high motion still holds eyes.
    m_motion_full = float(np.clip(all_frames_motion / PACING_MOTION_REF, 0.0, 1.0))
    m_pacing = 0.6 * m_cuts + 0.4 * m_motion_full

    m_audio_cons, rms_curve = audio_drop_penalty(audio, sr, duration)

    m_text = text_presence(all_frames)

    total = 100.0 * (0.40 * hook + 0.30 * m_pacing + 0.20 * m_audio_cons + 0.10 * m_text)

    subs = {
        "hook_motion_3s": round(m_hook_motion * 100, 1),
        "hook_audio_3s": round(m_hook_audio * 100, 1),
        "hook_brightness_3s": round(m_hook_bright * 100, 1),
        "cuts_total": n_cuts,
        "cuts_per_10s": round(cuts_per10, 2),
        "audio_consistency": round(m_audio_cons * 100, 1),
        "text_presence": round(m_text * 100, 1),
    }
    metrics = {
        "hook_3s": round(hook * 100, 1),
        "pacing": round(m_pacing * 100, 1),
        "audio_energy": round(m_audio_cons * 100, 1),
        "text": round(m_text * 100, 1),
        "total_score": round(total, 1),
    }

    suggestions = build_suggestions(subs, duration, m_pacing)

    return {
        "file": os.path.basename(path),
        "duration_s": round(duration, 2),
        "metrics": metrics,
        "submetrics": subs,
        "audio_rms_curve_1s": [round(x, 4) for x in rms_curve],
        "suggestions": suggestions,
        "note": "Heuristic scores from pixels + audio. Not neuroscience.",
    }


def build_suggestions(subs, duration, m_pacing):
    tips = []
    if subs["hook_motion_3s"] < 45:
        tips.append("Hook: first 1s has low movement — start with a fast visual, zoom, or pop so thumbs stop scrolling.")
    if subs["hook_audio_3s"] < 45:
        tips.append("Hook: audio starts quiet — hit a beat, SFX, or voice within the first second.")
    if subs["hook_brightness_3s"] < 35:
        tips.append("Hook: opening frames are dark — brighter first frame grabs more eyes.")
    if m_pacing < 0.35 and duration >= 10:
        tips.append(f"Pacing: only {subs['cuts_per_10s']} cuts/10s and low sustained motion — add a cut, zoom, or angle change every ~3s to hold attention.")
    if subs["audio_consistency"] < 60:
        tips.append("Audio: big energy drop-off detected mid-video — that's where viewers leave. Keep the beat/voice steady.")
    if subs["text_presence"] < 30:
        tips.append("Text: almost no on-screen text detected — bold captions help retention on mute.")
    if not tips:
        tips.append("Solid across all metrics — the hook, pacing, and audio all hold up. Ship it.")
    return tips[:3]


def print_report(res):
    m = res["metrics"]
    print(f"\n  Reel Hook Analyzer v1 — {res['file']} ({res['duration_s']}s)")
    print("  " + "=" * 52)
    print(f"  VIRALITY SCORE: {m['total_score']:.0f}/100")
    print("  " + "-" * 52)
    print(f"  Hook (first 3s)   {m['hook_3s']:5.1f}  [motion {res['submetrics']['hook_motion_3s']:.0f} | audio {res['submetrics']['hook_audio_3s']:.0f} | bright {res['submetrics']['hook_brightness_3s']:.0f}]")
    print(f"  Pacing            {m['pacing']:5.1f}  [{res['submetrics']['cuts_total']} cuts, {res['submetrics']['cuts_per_10s']}/10s]")
    print(f"  Audio energy      {m['audio_energy']:5.1f}")
    print(f"  Text presence     {m['text']:5.1f}")
    print("  " + "-" * 52)
    print("  Top suggestions:")
    for i, s in enumerate(res["suggestions"], 1):
        print(f"   {i}. {s}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Reel Hook Analyzer v1 — honest virality heuristics.")
    ap.add_argument("video", help="path to the reel/video file")
    ap.add_argument("--json", default=None, help="save JSON report path (default: <video>_hook_report.json)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    res = analyze(args.video)
    print_report(res)

    out = args.json or (os.path.splitext(args.video)[0] + "_hook_report.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  JSON saved: {out}")


if __name__ == "__main__":
    main()
