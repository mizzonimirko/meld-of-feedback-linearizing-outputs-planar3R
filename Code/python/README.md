# Real-time Python plots + planar 3R animation

This script reproduces the MATLAB-style layout:

- 7 animated time-series plots on the left,
- planar 3R robot animation on the right,
- active intervals as shaded backgrounds,
- active variables highlighted,
- active moving target shown correctly using `b1d`/`b2d`.

## Install

```bash
python -m pip install numpy scipy matplotlib
```

For MP4 export, install ffmpeg:

```bash
brew install ffmpeg
```

## Run interactively

```bash
python plot_3r_realtime.py path/to/simulink_3r_export.mat --realtime
```

## Save MP4

```bash
python plot_3r_realtime.py path/to/simulink_3r_export.mat --save-mp4 animated_vector_robot_python.mp4 --fps 60
```

## Faster preview

```bash
python plot_3r_realtime.py path/to/simulink_3r_export.mat --frame-step 5 --realtime
```
