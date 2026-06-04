# MELD 3R Robot Example

This repository contains the simulation, visualization, and compatibility-analysis code accompanying the numerical example of the paper on switching among multiple feedback-linearizing output selections, referred to as **melds**, for nonlinear control systems.

<img width="650" height="422" alt="meld_3r_robot_preview_650w_10fps" src="https://github.com/user-attachments/assets/bf0e5613-6a2b-4840-96fc-0867df7aaf24" />



The example considers a 3R planar robot equipped with two vacuum grippers. The robot is tasked with aspirating a sequence of items placed at different locations and different time instants. Some items are static, while others follow bounded time-varying trajectories between switching instants. The purpose of the example is to illustrate how a multi-dimensional deck of outputs can be switched through compatible melds while preserving boundedness guarantees.


## Overview

The robot has three revolute joints and a deck of seven candidate outputs:

- the three joint angles;
- the Cartesian coordinates of the first vacuum gripper;
- the Cartesian coordinates of the second vacuum gripper.

At each time interval, a three-dimensional output selection is activated. This active selection, or meld, contains the outputs needed to accomplish the current task, such as positioning one gripper above an item while regulating one joint coordinate. The simulation includes both static item references and bounded moving-item references to demonstrate tracking of time-varying desired output jets between switches.

The repository includes MATLAB/Simulink code for the closed-loop robot simulation, Python scripts for visualization, MATLAB scripts for paper-ready plots, and MATLAB scripts for sampled full-jet compatibility analysis.

## Repository Structure

    .
    ├── matlab/
    │   ├── desCmd.m
    │   ├── compatibility_analysis.m
    │   └── export_simulation_data.m
    │
    ├── python/
    │   ├── visualize_3r_rerun.py
    │   └── plot_3r_realtime.py
    │
    ├── data/
    │   └── simulink_3r_export.mat
    │
    ├── media/
    │   ├── meld_3r_robot_animation.gif
    │   └── meld_3r_robot_animation.mp4
    │
    ├── figures/
    │   ├── 3r_full_jet_all_outputs.pdf
    │   └── 3r_full_jet_max_inactive.pdf
    │
    └── README.md

The exact folder names can be adapted to the local organization of the project.

## Numerical Example

The simulation uses six scheduled intervals:

    [0,4], [4,7], [7,9], [9,13], [13,18], [18,22] seconds.

The corresponding melds are:

    sigma_1 = {q3, x_G1, y_G1}
    sigma_2 = {q2, x_G2, y_G2}
    sigma_3 = {q3, x_G1, y_G1}
    sigma_4 = {q3, x_G2, y_G2}
    sigma_5 = {q1, x_G2, y_G2}
    sigma_6 = {q1, q2, q3}

The second and third intervals include moving item references. These references follow bounded circular trajectories and therefore generate genuinely time-varying desired output jets. This directly illustrates the case where the reference signal changes continuously between switching instants.

## Compatibility Analysis

The repository includes a sampled computation of the full-jet compatibility errors associated with the deck of outputs. The desired full jet is composed of each desired output and its time derivative. The script reconstructs the state and velocity induced by the active desired outputs and compares the induced full deck with the desired full deck.

The sampled compatibility bound is computed as the maximum inactive-output full-jet mismatch over the simulation horizon. The exported plots show both all seven full-jet compatibility errors and the maximum inactive compatibility error.

## Running the Python Animation

After exporting the simulation data from MATLAB/Simulink, generate the MP4 animation with:

    python python/plot_3r_realtime.py data/simulink_3r_export.mat --save-mp4 media/meld_3r_robot_animation.mp4 --fps 60 --frame-step 1

For faster export, use:

    python python/plot_3r_realtime.py data/simulink_3r_export.mat --save-mp4 media/meld_3r_robot_animation.mp4 --fps 60 --frame-step 2

To preview the animation without saving:

    python python/plot_3r_realtime.py data/simulink_3r_export.mat --realtime

## Rerun Visualization

The repository also supports visualization through Rerun. If a recorded `.rrd` file and a saved viewer blueprint `.rbl` are provided, they can be opened in the Rerun Viewer to reproduce the visual layout used for the paper and video.

Suggested structure:

    rerun/
    ├── planar_3r_catching.rrd
    └── planar_3r_robot_view.rbl

Open the `.rrd` file in Rerun and load the `.rbl` blueprint to reproduce the saved viewer layout.

## Citation

If you use this code, please cite the associated paper.

    @article{MizzoniMelds2025,
      title   = {Switching among Feedback-Linearizing Output Selections for Nonlinear Systems},
      author  = {Mizzoni, Mirko and Coauthors},
      journal = {To appear},
      year    = {2025}
    }

Please replace the BibTeX entry with the final publication information once available.

## License

This repository is intended for academic and reproducibility purposes. Please add the appropriate license before public release.
