import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------
# Read single the data
# -----------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# Let's experiment with set 1
set_df = df[df["set"] == 1]

# 2D scatter plot
sns.scatterplot(data=set_df, x="acc_x", y="acc_y", hue="category")

# 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    df["acc_x"], df["acc_y"], df["acc_z"], c=pd.Categorical(df["category"]).codes
)

ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")

plt.suptitle("3D Scatter Plot")
plt.show()


# -----------------------------------------------------------------
# Adjust plot settings
# -----------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")

mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# -----------------------------------------------------------------
# Plot data for all exercises
# -----------------------------------------------------------------
"""
This section plots accelerometer and gyroscope data agaisn't the exercises labels,
to visualize their distribution and possible differences.
"""

columns = df.columns[:6]

for col in columns:
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))

    unique_labels = df["label"].unique()

    for idx, label in enumerate(unique_labels):
        subset = df[df["label"] == label]
        row = idx // 3  # Calculate the row index in the subplot grid
        col_index = idx % 3  # Calculate the column index in the subplot grid

        ax[row, col_index].plot(subset[col].reset_index(drop=True), label=label)
        ax[row, col_index].set_title(label, fontsize=30)

    plt.suptitle(col, fontsize=50, y=1.005)
    plt.tight_layout()  # Adjust layout to prevent overlapping titles
    plt.savefig(f"../../reports/figures/acc_gyr_comparion_per_label/{col}.png")
    plt.show()


# -----------------------------------------------------------------
# Compare medium vs heavy sets
# -----------------------------------------------------------------
"""
This section plots accelerometer and gyroscope data segmented by the exercise category
('heavy', 'medium', 'sitting', 'standing'), to visualize possible distribution differences
between medium and heavy exercises, which could be useful for a machine learning model
to differentiate sets.
"""

columns = df.columns[:6]

for col in columns:
    for participant in np.sort(df["participant"].unique()):
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))

        for idx, label in enumerate(df["label"].unique()):
            participant_category_df = (
                df.query(f"label == '{label}'")
                .query(f"participant == '{participant}'")
                .reset_index()
            )

            row = idx // 3  # Calculate the row index in the subplot grid
            col_index = idx % 3  # Calculate the column index in the subplot grid

            participant_category_df.groupby(["category"])[col].plot(
                ax=ax[row, col_index]
            )
            ax[row, col_index].set_title(label, fontsize=30)

            # Get the resulting labels (unique categories)
            handles, labels = ax[row, col_index].get_legend_handles_labels()

            # Set the legend with the resulting labels
            ax[row, col_index].legend(handles, labels, loc="upper right")

        plt.suptitle(
            f"Medium vs Heavy set | {participant} | {col}", fontsize=50, y=1.005
        )
        plt.tight_layout()

        plt.savefig(
            f"../../reports/figures/acc_gyr_comparison_medium_heavy/{col} ({participant}).png"
        )
        plt.show()


# -----------------------------------------------------------------
# Comparing participants
# -----------------------------------------------------------------

"""
This sections plots accelerometer and gyroscope data segmented by participant, to
visualize possible distribution differences between participants.
"""

columns = df.columns[:6]

for label in df["label"].unique():
    participants_df = (
        df.query(f"label == '{label}'").sort_values("participant").reset_index()
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))

    for i, col in enumerate(columns):
        row = i // 3  # Calculate the row index
        col_index = i % 3  # Calculate the column index

        participants_df.groupby(["participant"])[col].plot(ax=ax[row, col_index])
        ax[row, col_index].set_ylabel(f"{col}")
        ax[row, col_index].set_xlabel("samples")
        ax[row, col_index].set_title(col, fontsize=20, loc="right")
        ax[row, col_index].legend(loc="upper right")

    plt.suptitle(label, fontsize=30)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/acc_gyr_comparison_per_participant/{label}.png")
    plt.show()

columns = df.columns[:6]

for col in columns:
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))

    for idx, label in enumerate(df["label"].unique()):
        row = idx // 3  # Calculate the row index
        col_index = idx % 3  # Calculate the column index

        participants_df = (
            df.query(f"label == '{label}'").sort_values("participant").reset_index()
        )

        participants_df.groupby(["participant"])[col].plot(ax=ax[row, col_index])
        ax[row, col_index].set_ylabel(f"{col}")
        ax[row, col_index].set_xlabel("samples")
        ax[row, col_index].set_title(label, fontsize=20, loc="right")
        ax[row, col_index].legend(loc="upper right")

    plt.suptitle(col, fontsize=30)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/acc_gyr_comparison_per_participant/{col}.png")
    plt.show()

# -----------------------------------------------------------------
# Plot multiple axis
# -----------------------------------------------------------------

label = "squat"
participant = "A"

all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# -----------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# -----------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

# Accelerometer
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()

# Gyroscope
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()

# -----------------------------------------------------------------
# Combine plots in one figure
# -----------------------------------------------------------------

label = "row"
participant = "A"

combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=14,
)
ax[1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=14,
)
ax[1].set_xlabel("samples")

plt.suptitle(f"{label} ({participant})", fontsize=30, y=1.005)

# -----------------------------------------------------------------
# Loop over all combinations and export for both sensors
# -----------------------------------------------------------------

"""
This section plots a comparison between accelerometer and gyroscope distribution
agains't each exercise and participant.
"""

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=3,
                fancybox=True,
                shadow=True,
                fontsize=14,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=3,
                fancybox=True,
                shadow=True,
                fontsize=14,
            )
            ax[1].set_xlabel("samples")

            plt.suptitle(f"{label} ({participant})", fontsize=30, y=1.005)
            plt.savefig(
                f"../../reports/figures/acc_gyr_series/{label.title()} ({participant}).png"
            )
            plt.show()
