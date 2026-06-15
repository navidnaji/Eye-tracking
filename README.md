# Eye-tracking
Python pipeline for loading, cleaning, and analysing raw Tobii eye tracking data,detects fixations and saccades via velocity thresholding, labels Areas of Interest, and visualises gaze behaviour.
Tobii Eye Tracking Analysis Pipeline

A Python pipeline for loading, cleaning, and analysing raw eye tracking data exported from Tobii Pro Lab.

What It Does


Loads raw TSV/CSV data from Tobii Pro Lab
Cleans the data — removes track loss, interpolates short blinks
Calculates velocity between gaze samples
Classifies each sample as fixation or saccade using velocity thresholding (I-VT)
Labels AOIs — checks which Area of Interest each gaze point falls inside
Summarises results — dwell time per AOI, fixation/saccade ratio, first fixation
Visualises — gaze plot with AOI overlays, velocity over time plot
