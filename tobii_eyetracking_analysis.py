"""
Tobii Eye Tracking Data Analysis
=================================

Description:
    Loads raw Tobii eye tracking data, cleans it, detects fixations and saccades
    using velocity thresholding (I-VT), labels Areas of Interest (AOIs),
    and visualises the results.

Requirements:
    pip install pandas numpy matplotlib scipy

Usage:
    1. Export your Tobii session as a TSV file from Tobii Pro Lab
    2. Update the FILE_PATH and AOI definitions below
    3. Run: python tobii_eyetracking_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# =============================================================================
# CONFIGURATION — Edit these before running
# =============================================================================

FILE_PATH = "tobii_session.tsv"         # Path to your Tobii export file
SAMPLING_RATE_HZ = 120                  # Your tracker's sampling rate (e.g. 60, 120, 300)
VELOCITY_THRESHOLD = 30                 # Pixels/second — below = fixation, above = saccade
MIN_FIXATION_DURATION_MS = 100          # Minimum fixation duration in milliseconds
MIN_SACCADE_DURATION_MS = 20            # Minimum saccade duration in milliseconds
SCREEN_WIDTH = 1920                     # Screen resolution width in pixels
SCREEN_HEIGHT = 1080                    # Screen resolution height in pixels

# Define your Areas of Interest (AOIs)
# Format: "name": {"x1": left, "y1": top, "x2": right, "y2": bottom}
AOIS = {
    "top_left":     {"x1": 0,    "y1": 0,   "x2": 400,  "y2": 300},
    "centre":       {"x1": 660,  "y1": 290, "x2": 1260, "y2": 790},
    "bottom_right": {"x1": 1520, "y1": 780, "x2": 1920, "y2": 1080},
}

# Tobii column names — adjust if your export uses different names
COL_TIMESTAMP   = "RecordingTimestamp"
COL_GAZE_X      = "GazePointX(MCSpx)"
COL_GAZE_Y      = "GazePointY(MCSpx)"
COL_PUPIL_LEFT  = "PupilLeft"
COL_PUPIL_RIGHT = "PupilRight"
COL_VALIDITY    = "ValidityLeft"


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data(file_path):
    """Load raw Tobii TSV export into a DataFrame."""
    print(f"Loading data from: {file_path}")

    if not os.path.exists(file_path):
        print("File not found — generating sample data for demonstration...")
        return generate_sample_data()

    df = pd.read_csv(file_path, sep="\t")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def generate_sample_data(n_samples=1000):
    """Generate realistic sample Tobii data for demonstration purposes."""
    np.random.seed(42)
    timestamps = np.arange(0, n_samples * (1000 / 120), 1000 / 120)  # 120 Hz

    # Simulate fixations and saccades
    gaze_x = np.zeros(n_samples)
    gaze_y = np.zeros(n_samples)
    current_x, current_y = 960.0, 540.0

    for i in range(n_samples):
        if i % 100 == 0:  # saccade every ~100 samples
            current_x = np.random.uniform(100, 1820)
            current_y = np.random.uniform(100, 980)
        # Add small noise during fixation
        gaze_x[i] = current_x + np.random.normal(0, 5)
        gaze_y[i] = current_y + np.random.normal(0, 5)

    # Simulate some track loss (blinks)
    blink_indices = np.random.choice(n_samples, size=30, replace=False)
    validity = np.ones(n_samples)
    validity[blink_indices] = 0
    gaze_x[blink_indices] = np.nan
    gaze_y[blink_indices] = np.nan

    df = pd.DataFrame({
        COL_TIMESTAMP:  timestamps,
        COL_GAZE_X:     gaze_x,
        COL_GAZE_Y:     gaze_y,
        COL_PUPIL_LEFT: np.random.normal(3.2, 0.2, n_samples),
        COL_PUPIL_RIGHT:np.random.normal(3.2, 0.2, n_samples),
        COL_VALIDITY:   validity,
    })

    print(f"Generated {len(df)} sample rows at 120 Hz")
    return df


# =============================================================================
# STEP 2 — CLEAN DATA
# =============================================================================

def clean_data(df):
    """Remove invalid samples and interpolate short gaps (blinks)."""
    print("\nCleaning data...")
    total = len(df)

    # Mark invalid rows as NaN
    df.loc[df[COL_VALIDITY] == 0, [COL_GAZE_X, COL_GAZE_Y]] = np.nan

    # Count missing samples before interpolation
    missing = df[COL_GAZE_X].isna().sum()
    print(f"  Missing samples (track loss): {missing} ({missing/total*100:.1f}%)")

    # Interpolate short gaps only (e.g. blinks < 100ms = ~12 samples at 120Hz)
    max_gap_samples = int(0.1 * SAMPLING_RATE_HZ)
    df[COL_GAZE_X] = df[COL_GAZE_X].interpolate(method="linear", limit=max_gap_samples)
    df[COL_GAZE_Y] = df[COL_GAZE_Y].interpolate(method="linear", limit=max_gap_samples)

    # Drop remaining NaN rows (long gaps we don't interpolate)
    df = df.dropna(subset=[COL_GAZE_X, COL_GAZE_Y]).reset_index(drop=True)

    valid_rate = len(df) / total * 100
    print(f"  Valid data after cleaning: {len(df)} rows ({valid_rate:.1f}%)")

    if valid_rate < 80:
        print("  ⚠️  WARNING: Less than 80% valid data — consider excluding this session")

    return df


# =============================================================================
# STEP 3 — CALCULATE VELOCITY
# =============================================================================

def compute_velocity(df):
    """Calculate gaze velocity in pixels per second between consecutive samples."""
    dx = df[COL_GAZE_X].diff()
    dy = df[COL_GAZE_Y].diff()
    dt = df[COL_TIMESTAMP].diff() / 1000  # milliseconds → seconds

    distance = np.sqrt(dx**2 + dy**2)     # Euclidean distance (Pythagoras)
    velocity = distance / dt               # pixels per second

    df["velocity"] = velocity.fillna(0)
    return df


# =============================================================================
# STEP 4 — CLASSIFY FIXATIONS AND SACCADES (I-VT METHOD)
# =============================================================================

def classify_events(df):
    """
    Classify each sample as fixation, saccade, or blink using
    velocity thresholding (I-VT: Identification by Velocity Threshold).
    """
    print("\nClassifying fixations and saccades...")

    min_fix_samples = int(MIN_FIXATION_DURATION_MS / 1000 * SAMPLING_RATE_HZ)
    min_sac_samples = int(MIN_SACCADE_DURATION_MS  / 1000 * SAMPLING_RATE_HZ)

    # Initial classification by velocity
    df["event"] = np.where(df["velocity"] < VELOCITY_THRESHOLD, "fixation", "saccade")

    # Remove events shorter than minimum duration
    df = filter_short_events(df, "fixation", min_fix_samples)
    df = filter_short_events(df, "saccade",  min_sac_samples)

    # Summary
    fix_count = (df["event"] == "fixation").sum()
    sac_count = (df["event"] == "saccade").sum()
    print(f"  Fixation samples: {fix_count} ({fix_count/len(df)*100:.1f}%)")
    print(f"  Saccade samples:  {sac_count} ({sac_count/len(df)*100:.1f}%)")

    return df


def filter_short_events(df, event_type, min_samples):
    """Reclassify events shorter than the minimum duration as 'undefined'."""
    in_event = False
    start_idx = 0

    for i, row in df.iterrows():
        if row["event"] == event_type and not in_event:
            in_event = True
            start_idx = i
        elif row["event"] != event_type and in_event:
            duration = i - start_idx
            if duration < min_samples:
                df.loc[start_idx:i-1, "event"] = "undefined"
            in_event = False

    return df


# =============================================================================
# STEP 5 — LABEL AREAS OF INTEREST (AOIs)
# =============================================================================

def label_aois(df):
    """Label each gaze sample with the AOI it falls inside, or 'none'."""
    print("\nLabelling AOIs...")

    def get_aoi(row):
        for name, box in AOIS.items():
            if box["x1"] <= row[COL_GAZE_X] <= box["x2"] and \
               box["y1"] <= row[COL_GAZE_Y] <= box["y2"]:
                return name
        return "none"

    df["aoi"] = df.apply(get_aoi, axis=1)

    # AOI summary
    aoi_counts = df["aoi"].value_counts()
    print("  Gaze samples per AOI:")
    for aoi, count in aoi_counts.items():
        pct = count / len(df) * 100
        print(f"    {aoi:20s}: {count:5d} samples ({pct:.1f}%)")

    return df


# =============================================================================
# STEP 6 — SUMMARISE RESULTS
# =============================================================================

def summarise_results(df):
    """Print a summary of fixations and AOI dwell times."""
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)

    # Total recording duration
    duration_s = (df[COL_TIMESTAMP].max() - df[COL_TIMESTAMP].min()) / 1000
    print(f"\nRecording duration: {duration_s:.1f} seconds")

    # Fixation stats
    fixations = df[df["event"] == "fixation"]
    saccades  = df[df["event"] == "saccade"]
    print(f"\nFixations:  {len(fixations)} samples ({len(fixations)/len(df)*100:.1f}% of recording)")
    print(f"Saccades:   {len(saccades)} samples ({len(saccades)/len(df)*100:.1f}% of recording)")

    # AOI dwell time (how long gaze stayed in each AOI)
    print("\nAOI Dwell Time:")
    ms_per_sample = 1000 / SAMPLING_RATE_HZ
    for aoi in list(AOIS.keys()) + ["none"]:
        samples_in_aoi = (df["aoi"] == aoi).sum()
        dwell_ms = samples_in_aoi * ms_per_sample
        print(f"  {aoi:20s}: {dwell_ms:6.0f} ms ({dwell_ms/1000:.2f}s)")

    # First AOI looked at
    first_aoi = df[df["aoi"] != "none"]["aoi"].iloc[0] if len(df[df["aoi"] != "none"]) > 0 else "none"
    print(f"\nFirst AOI fixated: {first_aoi}")

    return df


# =============================================================================
# STEP 7 — VISUALISE
# =============================================================================

def visualise(df):
    """Plot gaze data with fixations, saccades, and AOI overlays."""
    print("\nGenerating visualisation...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Tobii Eye Tracking Analysis", fontsize=14, fontweight="bold")

    # --- Plot 1: Gaze Plot with AOIs ---
    ax1 = axes[0]
    ax1.set_title("Gaze Plot (Fixations & Saccades)")
    ax1.set_xlim(0, SCREEN_WIDTH)
    ax1.set_ylim(0, SCREEN_HEIGHT)
    ax1.invert_yaxis()  # Screen Y: 0 is at the top

    # Draw AOI rectangles
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    for i, (name, box) in enumerate(AOIS.items()):
        rect = patches.Rectangle(
            (box["x1"], box["y1"]),
            box["x2"] - box["x1"],
            box["y2"] - box["y1"],
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor=colors[i % len(colors)],
            alpha=0.15,
            label=f"AOI: {name}"
        )
        ax1.add_patch(rect)
        ax1.text(
            box["x1"] + 10, box["y1"] + 30,
            name, fontsize=9, color=colors[i % len(colors)], fontweight="bold"
        )

    # Plot fixations and saccades
    fix = df[df["event"] == "fixation"]
    sac = df[df["event"] == "saccade"]

    ax1.scatter(fix[COL_GAZE_X], fix[COL_GAZE_Y],
                s=8, alpha=0.4, color="#2196F3", label="Fixation", zorder=3)
    ax1.scatter(sac[COL_GAZE_X], sac[COL_GAZE_Y],
                s=4, alpha=0.4, color="#F44336", label="Saccade", zorder=3)

    ax1.set_xlabel("Screen X (pixels)")
    ax1.set_ylabel("Screen Y (pixels)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_facecolor("#F8F9FA")

    # --- Plot 2: Velocity Over Time ---
    ax2 = axes[1]
    ax2.set_title("Gaze Velocity Over Time")

    time_s = (df[COL_TIMESTAMP] - df[COL_TIMESTAMP].min()) / 1000
    ax2.plot(time_s, df["velocity"], color="#607D8B", linewidth=0.8, alpha=0.7, label="Velocity")
    ax2.axhline(y=VELOCITY_THRESHOLD, color="#F44336", linestyle="--",
                linewidth=1.5, label=f"Threshold ({VELOCITY_THRESHOLD} px/s)")

    ax2.fill_between(time_s, df["velocity"],
                     where=df["velocity"] < VELOCITY_THRESHOLD,
                     alpha=0.3, color="#2196F3", label="Fixation zone")
    ax2.fill_between(time_s, df["velocity"],
                     where=df["velocity"] >= VELOCITY_THRESHOLD,
                     alpha=0.3, color="#F44336", label="Saccade zone")

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Velocity (pixels/second)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_facecolor("#F8F9FA")
    ax2.set_ylim(0, min(df["velocity"].quantile(0.99) * 1.2, 2000))

    plt.tight_layout()
    output_path = "eyetracking_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {output_path}")
    plt.show()


# =============================================================================
# MAIN — Run the full pipeline
# =============================================================================

def main():
    print("=" * 50)
    print("TOBII EYE TRACKING ANALYSIS PIPELINE")
    print("=" * 50)

    # Run each step in order
    df = load_data(FILE_PATH)
    df = clean_data(df)
    df = compute_velocity(df)
    df = classify_events(df)
    df = label_aois(df)
    df = summarise_results(df)
    visualise(df)

    # Save processed data
    output_csv = "eyetracking_processed.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nProcessed data saved to: {output_csv}")
    print("\nDone!")


if __name__ == "__main__":
    main()
