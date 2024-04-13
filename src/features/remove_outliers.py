# --------------------------------------------------------------
# Importing necessary libraries
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson

from colorama import Fore, Back, Style


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
outlier_columns = list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

df[["acc_x", "label"]].boxplot(by="label", figsize=(20, 10))
plt.show()

df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))
plt.show()


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """
    Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["no outlier " + col, "outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# Check for normal distribution

# Histogram plot
df[outlier_columns[:3] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)
df[outlier_columns[3:] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)
plt.show()

for col in outlier_columns:
    df[col].plot.hist(figsize=(20, 20), layout=(3, 3))
    plt.title(col)
    plt.show()

# Quantile-quantile plot
import statsmodels.api as sm

for col in outlier_columns:
    sm.qqplot(df[col], line="45")
    plt.title(col)
    plt.show()

# Statistical tests

"""
If the p-value is “small” - that is, if there is a low probability
of sampling data from a normally distributed population that produces such an extreme 
value of the statistic - this may be taken as evidence against the null hypothesis in 
favor of the alternative: the weights were not drawn from a normal distribution.
"""

# Shapiro Test


def shapiro_test(data):
    alpha = 0.05
    for col in data.columns:
        stat, p = shapiro(df[col])

        print(Style.RESET_ALL)
        print(f"\nColumn: {col} \nStatistics={stat:.3f}, p={p:.5f}")

        if p > alpha:
            print(Fore.GREEN + "Sample looks Gaussian (fail to reject H0)")
        else:
            print(Fore.RED + "Sample does not look Gaussian (reject H0)")


shapiro_test(df[outlier_columns])

# D’Agostino’s K^2 Test


def dagostino_test(data):
    alpha = 0.05
    for col in data.columns:
        stat, p = normaltest(df[col])

        print(Style.RESET_ALL)
        print(f"\nColumn: {col} \nStatistics={stat:.3f}, p={p:.5f}")

        if p > alpha:
            print(Fore.GREEN + "Sample looks Gaussian (fail to reject H0)")
        else:
            print(Fore.RED + "Sample does not look Gaussian (reject H0)")


dagostino_test(df[outlier_columns])

# Anderson-Darling Test

for col in outlier_columns:
    result = anderson(df[col])
    print(f"\nColumn: {col} \nStatistics={result.statistic:.3f}")
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]

        if result.statistic < result.critical_values[i]:
            print("%.3f: %.3f, data looks normal (fail to reject H0)" % (sl, cv))
        else:
            print("%.3f: %.3f, data does not look normal (reject H0)" % (sl, cv))


"""
According to the tests performed the data does not seem to be normally distributed,
thus when using a method to identify and remove outliers, we should use something that
does not assume normality in the data.
"""

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Insert IQR function


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# --------------------------------------------------------------
# Local Outlier Factor
# --------------------------------------------------------------

# Insert LOF function


def mark_outliers_lof(dataset, col, n_neighbors, contamination=0.1):
    """Function to mark values as outliers using the Local Outlier Factor method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        contamination (float): The percentage of values to be considered as outliers (from 0.1 to 0.5)
        n_neighbors (integer): The number of neighbor points the method should consider to evaluate the LOF

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    clf.fit_predict(dataset[col].to_numpy().reshape(-1, 1))

    results = clf.negative_outlier_factor_

    dataset["LOF"] = results.tolist()

    dataset[col + "_outlier"] = dataset["LOF"] < -1.5

    return dataset


# --------------------------------------------------------------
# Isolation Forest
# --------------------------------------------------------------

# Insert isolation forest function


def mark_outliers_isolation_forest(dataset, col, contamination=0.1, random_state=42):
    """Function to mark values as outliers using the Isolation Forest method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        contamination (float): The percentage of values to be considered as outliers (from 0.1 to 0.5)
        random_state (integer): Used to initialize the random number generator to ensure reproducibility of the results

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    reshaped_col = dataset[col].to_numpy().reshape(-1, 1)

    model_IF = IsolationForest(contamination=contamination, random_state=random_state)
    model_IF.fit(reshaped_col)

    dataset["anomaly_score"] = model_IF.decision_function(reshaped_col)

    dataset[col + "_outlier"] = model_IF.predict(reshaped_col)

    dataset[col + "_outlier"] = dataset[col + "_outlier"] == -1

    return dataset


# Plot a single column

col = "acc_x"
dataset = mark_outliers_iqr(df, col)

plot_binary_outliers(
    dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
)

# Loop over all columns

# IQR
for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)

    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# LOF
for col in outlier_columns:
    dataset = mark_outliers_lof(df, col, n_neighbors=150)

    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# Isolation Forest
for col in outlier_columns:
    dataset = mark_outliers_isolation_forest(df, col)

    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

label = "squat"

for col in outlier_columns:
    dataset = mark_outliers_lof(df[df["label"] == label], col, n_neighbors=35)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column
col = "gyr_z"
dataset = mark_outliers_lof(df, col, n_neighbors=35)
dataset[dataset["gyr_z_outlier"]]

dataset.loc[dataset["gyr_z_outlier"], col] = np.nan

# Create a function to loop over all columns and labels and remove outliers


def remove_outliers_from_df(dataset, outlier_columns, n_neighbors):
    """
    Function to remove outliers from the original dataset, which were detected using
    the mark_outliers_lof() function.

    Args:
        dataset (pd.DataFrame): The dataset in which outliers should be removed
        outlier_columns (list of string): A list with the name of the numerical columns which might contain outliers
        n_neighbors (integer): The number of neighbor points the method should consider to evaluate the LOF

    Returns:
        tuple: A tuple containing two elements:
            - outliers_removed_total (integer): The total amount of outliers removed
            from the original dataset given the input parameters.
            - outliers_removed_df (pd.DataFrame): A dataframe with the outlier values set
            to NaN.
    """

    outliers_removed_df = df.copy()
    outliers_removed_total = 0

    for col in outlier_columns:
        for label in df["label"].unique():
            dataset = mark_outliers_lof(
                df[df["label"] == label], col, n_neighbors=n_neighbors
            )

            # Replace values marked as outliers with NaN
            dataset.loc[dataset[col + "_outlier"], col] = np.nan

            # Update the column in the original dataframe
            outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = (
                dataset[col]
            )

            n_outliers = len(df) - len(outliers_removed_df[col].dropna())
            outliers_removed_total = outliers_removed_total + n_outliers

    return outliers_removed_total, outliers_removed_df


# Evaluate the amount of outliers removed for different values of 'n_neighbors' and append to a dictionary

outliers_removed = {}
for n in range(1, 100):  # n is the value of the n_neighbors parameter
    total_outliers, outliers_removed_df = remove_outliers_from_df(
        df, outlier_columns, n
    )

    print(f"n:{n}, total outliers removed: {total_outliers}")

    outliers_removed.update({n: total_outliers})


def plot_from_dict(dictionary):
    """
    Function to plot the data from a dictionary and highlight the smallest value.

    Args:
        dictionary (dict): A dictionary containing x and y values for plotting.

    Returns:
        matplotlib.figure.Figure: A matplotlib figure object representing the plot.

    This function takes a dictionary containing x and y values (keys and values, respectively)
    and creates a plot to visualize the data. It plots the y values against the x values and
    highlights the smallest value with a red circle and an annotation. The plot includes a title,
    axis labels, grid lines, and a legend.
    """

    x_values = list(dictionary.keys())
    y_values = list(dictionary.values())

    min_y_value = min(y_values)
    min_x_value = x_values[y_values.index(min_y_value)]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker="o", linestyle="-", label="Data")
    plt.scatter(
        min_x_value, min_y_value, color="indianred", label="Smallest Value", s=100
    )
    plt.annotate(
        f"Smallest Value:\n{min_y_value} for n = {min_x_value}",
        xy=(min_x_value, min_y_value),
        xytext=(min_x_value + 1, min_y_value + 3000),
        fontsize=15,
        color="indianred",
        arrowprops=dict(arrowstyle="->", color="indianred", linewidth=3),
    )
    plt.title("Number of Outliers Removed")
    plt.xlabel("N Neighbors")
    plt.ylabel("Total Outliers Removed")
    plt.grid(True)
    plt.legend()
    plt.show()


plot_from_dict(outliers_removed)

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_lof.pkl")
