# Reel Hook Analyzer

Scores a short video's viral potential with **honest, code-measured heuristics**.
No fake "brain scan" claims — everything here is computed from pixels and audio with ffmpeg + numpy.

## What it measures (v1)

| Metric | Weight | How |
|---|---|---|
| **Hook (first 3s)** | 40% | Motion intensity (frame diff), audio energy, brightness punch in the opening 3 seconds |
| **Pacing** | 30% | Cuts per 10 seconds via ffmpeg scene detection |
| **Audio energy** | 20% | Biggest drop-off between 1-second RMS windows (drop-offs = where viewers leave) |
| **Text presence** | 10% | On-screen text likelihood via an edge-density heuristic (crude in v1) |

Output: a 0–100 virality score, per-metric breakdown, and the top 3 concrete suggestions (e.g. "hook has low movement in the first second").

## Honesty note

These are **editing heuristics**, not neuroscience. A high score means the video *looks and sounds* like high-retention reels (fast hook, steady energy, good pacing). It cannot predict actual virality — the algorithm, topic, and luck matter more. Use it as a pre-post checklist, not a crystal ball.

## Usage

```bash
pip install -r requirements.txt   # numpy only; ffmpeg must be installed separately
python analyzer.py myreel.mp4
python analyzer.py myreel.mp4 --json report.json
```

## Calibration

Reference ranges are fixed constants at the top of `analyzer.py`
(`MOTION_REF`, `PACING_MOTION_REF`, `AUDIO_REF`, `EDGE_REF`, `CUTS_REF`, `SCENE_THRESH`).
`MOTION_REF` was set empirically: the 90th percentile of frame-to-frame motion
in lively code-animation reels measured ~7.5 (320×180 @ 10fps).
Pacing blends hard cuts (60%) with sustained motion (40%) so one-shot
animations without cuts aren't scored zero.
Tune them against your own winning videos — the defaults are sane starting points, not gospel.

## License

MIT — see `LICENSE`.
