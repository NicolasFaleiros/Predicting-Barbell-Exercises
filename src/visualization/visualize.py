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

    plt.suptitle(col, fontsize=50, y=1.02)
    plt.tight_layout()  # Adjust layout to prevent overlapping titles
    plt.show()

# -----------------------------------------------------------------
# Compare medium vs heavy sets
# -----------------------------------------------------------------
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
                ax=ax[row, col_index], label=label
            )
            ax[row, col_index].set_title(label, fontsize=30)

        plt.suptitle(
            f"Medium vs Heavy set | {participant} | {col}", fontsize=50, y=1.02
        )
        plt.tight_layout()  # Adjust layout to prevent overlapping titles
        plt.legend()
        plt.show()


# -----------------------------------------------------------------
# Comparing participants
# -----------------------------------------------------------------

acc_col = ["acc_x", "acc_y", "acc_z"]

for label in df["label"].unique():
    participants_df = (
        df.query(f"label == '{label}'").sort_values("participant").reset_index()
    )

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 12))

    for i, col in enumerate(acc_col):
        participants_df.groupby(["participant"])[col].plot(ax=ax[i])
        ax[i].set_ylabel(f"{col}")
        ax[i].set_xlabel("samples")
        ax[i].set_title(col, fontsize=20, loc="right")
        ax[i].legend()

    plt.suptitle(label, fontsize=30)

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

# -----------------------------------------------------------------
# Loop over all combinations and export for both sensors
# -----------------------------------------------------------------

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

            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()


def plot_highest_highest(dataset):
    plt.figure(figsize=(15, 15))
    ax = sns.barplot(
        x="model",
        y="accuracy",
        hue="feature_set",
        data=dataset,
        palette=["lightgrey"],
        edgecolor="darkgrey",
    )

    categories = dataset["model"].unique()

    # Calculate the maximum height of the bars
    max_bar_height = max([patch.get_height() for patch in ax.patches])

    # Set the annotation height as the maximum bar height
    annotation_height = max_bar_height

    for i, patch in enumerate(ax.patches):
        # Get the index of the category corresponding to the current bar
        category_index = i % len(categories)
        # Get the categorical value associated with the current bar
        category = categories[category_index]

        # Extract the x and y values of the current bar
        x_value = patch.get_x() + patch.get_width() / 2
        y_value = patch.get_height()

        # Find the highest and second highest y values for the current category
        highest_y = dataset[(dataset["model"] == category)]["accuracy"].max()

        second_highest_y = dataset[
            (dataset["model"] == category) & (dataset["accuracy"] < highest_y)
        ]["accuracy"].max()

        feature_set = dataset["feature_set"].iloc[i]

        cluster_center_x = ax.patches[i].get_x() + ax.patches[i].get_width() / 2

        # Set the color of the bar based on its y value
        if y_value == highest_y:
            patch.set_color(
                "#344C11"
            )  # Highlight the bar with the highest y value in green
            ax.annotate(
                f"{feature_set}\n{y_value:.2f}",
                (cluster_center_x, annotation_height + 0.015),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="#344C11",
            )
        elif y_value == second_highest_y:
            patch.set_color(
                "#778D45"
            )  # Highlight the bar with the second-highest y value in lighter green
            ax.annotate(
                f"{feature_set}\n{y_value:.2f}",
                (cluster_center_x, annotation_height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="#778D45",
            )

        # elif y_value == lowest_y:
        #    patch.set_color("indianred") # Highlight the bar with the lowest y value in lighter green

    ax.legend_.remove()

    plt.suptitle("Highest and second-highest accuracy for each algorithm", fontsize=20)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1.02)
    plt.show()


def plot_highest_lowest(dataset):
    plt.figure(figsize=(15, 15))
    ax = sns.barplot(
        x="model",
        y="accuracy",
        hue="feature_set",
        data=dataset,
        palette=["lightgrey"],
        edgecolor="darkgrey",
    )

    categories = dataset["model"].unique()

    # Calculate the maximum height of the bars
    max_bar_height = max([patch.get_height() for patch in ax.patches])

    # Set the annotation height as the maximum bar height
    annotation_height = max_bar_height

    for i, patch in enumerate(ax.patches):
        # Get the index of the category corresponding to the current bar
        category_index = i % len(categories)
        # Get the categorical value associated with the current bar
        category = categories[category_index]

        # Extract the x and y values of the current bar
        x_value = patch.get_x() + patch.get_width() / 2
        y_value = patch.get_height()

        # Find the highest y value for the current category
        highest_y = dataset[(dataset["model"] == category)]["accuracy"].max()

        # Find the highest and second lowest y value for the current category
        lowest_y = dataset[(dataset["model"] == category)]["accuracy"].min()

        feature_set = dataset["feature_set"].iloc[i]

        cluster_center_x = ax.patches[i].get_x() + ax.patches[i].get_width() / 2

        # Set the color of the bar based on its y value
        if y_value == highest_y:
            patch.set_color(
                "#344C11"
            )  # Highlight the bar with the highest y value in green
            ax.annotate(
                f"{feature_set}\n{y_value:.2f}",
                (cluster_center_x, annotation_height + 0.015),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="#344C11",
            )

        elif y_value == lowest_y:
            patch.set_color(
                "indianred"
            )  # Highlight the bar with the lowest y value in lighter green
            ax.annotate(
                f"{feature_set}\n{y_value:.2f}",
                (cluster_center_x, annotation_height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="indianred",
            )

    ax.legend_.remove()

    plt.suptitle("Highest and lowest accuracy for each algorithm", fontsize=20)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1.02)
    plt.show()
