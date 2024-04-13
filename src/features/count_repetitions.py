import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_transformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]  # We can't count repetitions from "rest sets"

acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = squat_df
plot_set = plot_df["set"].unique()[1]
acc_columns = [col for col in plot_df.columns if "acc" in col]
gyr_columns = [col for col in plot_df.columns if "gyr" in col]
columns_list = [acc_columns, gyr_columns]


# Create a figure and a grid of subplots (2 rows, 2 columns)
for columns in columns_list:
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    for i in range(0, 2):
        for j in range(0, 2):
            column_name = columns[i * 2 + j]
            ax[i, j].plot(plot_df[plot_df["set"] == plot_set][column_name])
            ax[i, j].set_title(f"{column_name}")

    # Add a title to the entire figure
    fig.suptitle(f"Accelerometer data for set {plot_set}", fontsize=50)

    # Show the plot
    plt.show()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = int(1000 / 200)  # 5 instances per second in the data
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

bench_set["acc_r"].plot()
col = "acc_r"

LowPass.low_pass_filter(
    bench_set, col=col, sampling_frequency=fs, cutoff_frequency=0.4, order=5
)[col + "_lowpass"].plot()


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
def count_reps(dataset, fs, cutoff=0.4, order=10, column="acc_r", plot=True):
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )

    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]

    if plot:
        fig, ax = plt.subplots()
        plt.plot(dataset[f"{column}_lowpass"], color="royalblue")
        plt.plot(peaks[f"{column}_lowpass"], "o", color="indianred")
        ax.set_ylabel(f"{column}_lowpass")
        exercise = dataset["label"].iloc[0].title()
        category = dataset["category"].iloc[0].title()
        plt.title(f"{category} {exercise}: {len(peaks)} Reps")
        plt.show()

    return len(peaks)


count_reps(bench_set, fs=5, cutoff=0.4)
count_reps(squat_set, fs=5, cutoff=0.35)
count_reps(row_set, fs=5, cutoff=0.65)
count_reps(ohp_set, fs=5, cutoff=0.35)
count_reps(dead_set, fs=5, cutoff=0.4)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)

rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()
rep_df["reps_pred"] = 0

for s in df["set"].unique():
    subset = df[df["set"] == s]

    column = "acc_r"
    cutoff = 0.4

    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35

    if subset["label"].iloc[0] == "row":
        cutoff = 0.65
        column = "gyr_x"

    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.35

    reps = count_reps(subset, cutoff=cutoff, column=column, fs=5, plot=False)

    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

rep_df

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)

print(f"MAE: {error}")

rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().plot.bar()
plt.xticks(rotation=45)
