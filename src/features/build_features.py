import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_transformation import LowPassFilter, PrincipalComponentAnalysis
from temporal_abstraction import NumericalAbstraction
from frequency_abstraction import FourierTransformation

from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data and overall settings
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_lof.pkl")

predictor_columns = list(df.columns[:6])

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()

# --------------------------------------------------------------
# Calculating average set duration
# --------------------------------------------------------------

for s in df["set"].unique():

    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start

    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5  # How long each repetition of a heavy set lasted
duration_df[1] / 10  # How long each repetition of a medium set lasted

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()  # A technique to "smooth" the data

fs = 1000 / 200  # The frequency of the data we're working with (1s / 200ms)
cutoff = 1.3  # Optimal value found after try out

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Principal component number")
plt.ylabel("Explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 18]

subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()

predictor_columns = list(df.columns[:6])
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000 / 200)  # Window size of 5 (1 second, since data was recorded every 200ms)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s]

    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")

    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

subset[["acc_x", "acc_x_temp_mean_ws_5", "acc_x_temp_std_ws_5"]].plot()
plt.show()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()

FreqAbs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)  # Average length of a repetition (2.8 s)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# Ploting an example of the Discrete Fourier Transformation
subset = df_freq[df_freq["set"] == 35]
subset[
    [
        "acc_y",
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot(figsize=(20, 10))
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna(subset=["gyr_r_freq_weighted"])
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

# Clustering with the accelerometer data

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plotting the results for the accelerometer cluster

# Create a figure with subplots
fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(25, 20), subplot_kw={"projection": "3d"}
)

# Plot data on the first subplot (ax[0])
for c in np.sort(df_cluster["cluster"].unique()):
    subset = df_cluster[df_cluster["cluster"] == c]
    ax[0].scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax[0].set_xlabel("X axis")
ax[0].set_ylabel("Y axis")
ax[0].set_zlabel("Z axis")
ax[0].set_title("Accelerometer data by cluster")
ax[0].legend()

# Plot data on the second subplot (ax[1])
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax[1].scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax[1].set_xlabel("X axis")
ax[1].set_ylabel("Y axis")
ax[1].set_zlabel("Z axis")
ax[1].set_title("Accelerometer data by label")
ax[1].legend()

plt.show()

# Clustering with the gyroscope data

df_cluster = df_freq.copy()

cluster_columns = ["gyr_x", "gyr_y", "gyr_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=6, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plotting the results for the gyroscope cluster

# Create a figure with subplots
fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(25, 20), subplot_kw={"projection": "3d"}
)

# Plot data on the first subplot (ax[0])
for c in np.sort(df_cluster["cluster"].unique()):
    subset = df_cluster[df_cluster["cluster"] == c]
    ax[0].scatter(subset["gyr_x"], subset["gyr_y"], subset["gyr_z"], label=c)
ax[0].set_xlabel("X axis")
ax[0].set_ylabel("Y axis")
ax[0].set_zlabel("Z axis")
ax[0].set_title("Gyroscope data by cluster")
ax[0].legend()

# Plot data on the second subplot (ax[1])
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax[1].scatter(subset["gyr_x"], subset["gyr_y"], subset["gyr_z"], label=l)
ax[1].set_xlabel("X axis")
ax[1].set_ylabel("Y axis")
ax[1].set_zlabel("Z axis")
ax[1].set_title("Gyroscope data by label")
ax[1].legend()

plt.show()

"""
It seems that the clusters generated using the accelerometer data fits well with the
actual accelerometer data separated by label. On the other hand, the gyroscope data
when plotted by label seems like a big mess, so we don't have an idea of how good are
the clusters from the gyroscope data.
"""

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
