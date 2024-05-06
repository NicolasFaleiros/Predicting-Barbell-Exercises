"""
This script aims to provide functions that will turn the visualization and EDA process easier. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100


def check_distribution(dataset, columns, color="#003049"):
    """
    Function to plot a histogram graph with a kernel density estimate. It will plot
    as much charts as the length of the 'columns' list.

    Args:
        dataset (pd.DataFrame): The dataset with the numerical values.
        columns (list): A list of column names representing the numerical variables for the histogram.
        color (str, optional): A color for the charts (preferably a hex code).

    Returns:
        None

    Example:
        columns = list(dataset.columns[:6])
        check_distribution(dataset=df, columns=columns, color="#003049")
    """

    num_cols = len(columns)
    num_rows = num_cols // 3 + (num_cols % 3 > 0)

    fig, ax = plt.subplots(num_rows, 3, figsize=(24, 12))

    for i, feature in enumerate(columns):
        row = i // 3
        col = i % 3

        sns.histplot(
            data=dataset,
            x=feature,
            kde=True,
            ax=ax[row, col],
            stat="proportion",
            color=color,
        )

        ax[row, col].set_ylabel("")


def check_distribution_by_label(dataset, label, colors=None, **kwargs):
    """
    Function to plot a histogram graph with a kernel density estimate. It will plot
    as much charts as the length of the 'columns' list.

    Args:
        dataset (pd.DataFrame): The dataset with the numerical values.
        label (str): The dataset column name to be used as label for the charts.
        colors (str, optional): A list of colors, one for each histogram plotted (preferably a hex code).
        **kwargs: Additional keyword arguments:
            columns (list): A list of column names representing the numerical variables for the histogram.

    Returns:
        None

    Example:
        outlier_columns = list(dataset.columns[:6])
        acc = outlier_columns[:3]
        gyr = outlier_columns[3:]
        colors = ["#003049", "#386641", "#C1121F"]
        check_distribution_by_label(dataset=df, label="label", colors=colors, acc=acc, gyr=gyr)
    """

    length = []
    outliers_cols = ()

    for key, value in kwargs.items():
        col = value
        outliers_cols += (col,)

        if isinstance(value, list):
            length.append(len(value))

    if colors is None:
        num_features = max(length)
        default_colors = [f"C{i}" for i in range(num_features)]
        colors = default_colors

    label_cols = dataset[label].unique().tolist()

    num_cols = len(label_cols)  # Número de variáveis
    num_rows = num_cols // 3 + (num_cols % 3 > 0)

    for features_index in range(0, len(outliers_cols)):
        fig, ax = plt.subplots(num_rows, 3, figsize=(24, 12))

        for i, labels in enumerate(label_cols):
            row = i // 3
            col = i % 3

            for j, feature in enumerate(outliers_cols[features_index]):
                sns.histplot(
                    data=dataset[dataset[label] == labels],
                    x=feature,
                    kde=True,
                    ax=ax[row, col],
                    stat="proportion",
                    color=colors[j],
                    label=feature,
                )

            ax[row, col].set_title(f"{labels}", fontsize=30, y=1.005)
            ax[row, col].set_xlabel("")
            ax[row, col].set_ylabel("")

        handles, labels = ax[0, 0].get_legend_handles_labels()

        # Create legend
        fig.legend(
            handles,
            labels,
            loc="upper center",
            fontsize=20,
            bbox_to_anchor=(0.5, 1.15),
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout()
        plt.show()


def plot_highest_highest(dataset, x, y, hue):
    """
    Function to plot a bar graph in which the x-axis is a categorical variable,
    the y-axis is a numerical variable and the hue is a categorical variable.
    The bars associated with the two highest values in each section will be highlighted
    in shades of green, with proper annotations.

    Args:
        dataset (pd.DataFrame): The dataset
        x (str): The column name representing the categorical variable for the x-axis.
        y (str): The column name representing the numerical variable for the y-axis.
        hue (str): The column name representing the categorical variable for the hue.

    Returns:
        None
    """

    plt.figure(figsize=(15, 15))
    ax = sns.barplot(
        x=x,
        y=y,
        hue=hue,
        data=dataset,
        palette=["lightgrey"],
        edgecolor="darkgrey",
    )

    categories = dataset[x].unique()

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
        highest_y = dataset[(dataset[x] == category)][y].max()

        second_highest_y = dataset[(dataset[x] == category) & (dataset[y] < highest_y)][
            y
        ].max()

        feature_set = dataset[hue].iloc[i]

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

    ax.legend_.remove()

    plt.suptitle("Highest and second-highest accuracy for each algorithm", fontsize=20)
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.ylim(0.7, 1.02)
    plt.show()


def plot_highest_lowest(dataset, x, y, hue):
    """
    Function to plot a bar graph in which the x-axis is a categorical variable,
    the y-axis is a numerical variable and the hue is a categorical variable.
    The bars associated with the highest value in each section will be highlighted
    in green, while the bars associated with the lowest values will be highlighted
    in red, with proper annotations.

    Args:
        dataset (pd.DataFrame): The dataset
        x (str): The column name representing the categorical variable for the x-axis.
        y (str): The column name representing the numerical variable for the y-axis.
        hue (str): The column name representing the categorical variable for the hue.

    Returns:
        None
    """

    plt.figure(figsize=(15, 15))
    ax = sns.barplot(
        x=x,
        y=y,
        hue=hue,
        data=dataset,
        palette=["lightgrey"],
        edgecolor="darkgrey",
    )

    categories = dataset[x].unique()

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
        highest_y = dataset[(dataset[x] == category)][y].max()

        # Find the highest and second lowest y value for the current category
        lowest_y = dataset[(dataset[x] == category)][y].min()

        feature_set = dataset[hue].iloc[i]

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
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.ylim(0.7, 1.02)
    plt.show()
