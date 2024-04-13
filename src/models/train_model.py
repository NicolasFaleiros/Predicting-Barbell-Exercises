import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from learning_algorithms import ClassificationAlgorithms

import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "visualization", "visualize")))
from visualize import plot_highest_highest, plot_highest_lowest

import itertools


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Import the data

df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set"], axis=1)

X = df_train.drop("label", axis=1)
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Plot to see the distribution of each target class in the y_train and y_test sets
fig, ax = plt.subplots(figsize=(10, 5))
y_train.value_counts().plot(kind="bar", ax=ax, color="lightblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
frequency_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
cluster_features = ["cluster"]

features = (
    ("basic_features", basic_features),
    ("square_features", square_features),
    ("pca_features", pca_features),
    ("time_features", time_features),
    ("frequency_features", frequency_features),
    ("cluster_features", cluster_features),
)

for feature_name, feature_list in features:
    print(f"{feature_name}: {len(feature_list)}")

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()
max_features = 10

"""
This piece of code may take a while to run !!!

Using a decision tree model, the forward selection loops over all of 
our features and detects the best performing one. Then it loops all 
over again, to train the model without the best performing feature, 
then detect the second best, and so on, until it reaches max_features 
number of best features. From this proccess we should expect diminushing
results, which means that after some point, adding another relevant 
feature to the training proccess will not give us a significant increase 
of accuracy.

Also note that we are training the model using the training data, and also
evaluating the performance using the train data. In some sense we are 
"cheating", but it gives us a sense of direction to determine if our 
efforts of feature engineering were paid off.

As it takes a while to run this piece of code I will leave my results
stored in the `selected_features`, but be aware that there might be
some kind of stochastic proccess running in the background and if you
run the code yourself you could get slightly different results.
"""

# Forward selection algorithm
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

# My result for the 10 best features with forward selection
selected_features = [
    "acc_y_temp_mean_ws_5",
    "gyr_r_freq_0.0_Hz_ws_14",
    "duration",
    "acc_z_freq_0.0_Hz_ws_14",
    "gyr_r_freq_2.143_Hz_ws_14",
    "acc_y_freq_0.0_Hz_ws_14",
    "acc_z_max_freq",
    "gyr_z_freq_1.429_Hz_ws_14",
    "gyr_x_temp_std_ws_5",
    "acc_x_freq_0.714_Hz_ws_1",
]

# Diminushing returns of accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "feature_set_1",
    "feature_set_2",
    "feature_set_3",
    "feature_set_4",
    "selected_features",
]

iterations = 1
score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

score_df.sort_values(by="accuracy", ascending=False, inplace=True)

plt.figure(figsize=(15, 15))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(
    loc="center right",
    bbox_to_anchor=(1.25, 0.5),
    fancybox=True,
    shadow=True,
    fontsize=18,
)

plot_highest_highest(score_df)
plot_highest_lowest(score_df)


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

"""
As the model who had the highest accuracy was the Feedforward Neural Network, using
the `feature_set_4` set of features, I will begin by evaluating it a bit more closely,
and also retrain the model using gridsearch.
"""
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

plt.figure(figsize=(15, 15))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
)
plt.title("Confusion Matrix")
plt.xlabel("Predict label")
plt.ylabel("True label")
plt.yticks(rotation=360)

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

participant_df = df.drop(["set", "category"], axis=1)

X_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]

X_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
y_test = participant_df[participant_df["participant"] == "A"]["label"]

X_train = X_train.drop(["participant"], axis=1)
X_test = X_test.drop(["participant"], axis=1)

# Plot to see the distribution of each target class in the y_train and y_test sets
fig, ax = plt.subplots(figsize=(10, 5))
y_train.value_counts().plot(kind="bar", ax=ax, color="lightblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

"""
Before we were training and evaluating the model using data from all participants,
which could be causing some data leakage, as the model could be evaluating its 
performance on a given observation in the test set and nourishing good returns as
it had already trained using data from that particular participant in the training set.

Now we are going to train the model again but we will do it without using any data
from participant `A`, and then we will evaluate the model solely using features from
this participant.

We are doing this, of course, to ensure that our model is able to generalize its 
good prediction capabilities to really new data (as if a new participant appeared to 
use our fitness tracker device).
"""
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

plt.figure(figsize=(15, 15))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
)
plt.title("Confusion Matrix")
plt.xlabel("Predict label")
plt.ylabel("True label")
plt.yticks(rotation=360)

# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------

(class_train_y, class_test_y, class_train_prob_y, class_test_prob_y) = (
    learner.decision_tree(
        X_train[selected_features], y_train, X_test[selected_features], gridsearch=False
    )
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

plt.figure(figsize=(15, 15))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
)
plt.title("Confusion Matrix")
plt.xlabel("Predict label")
plt.ylabel("True label")
plt.yticks(rotation=360)

# --------------------------------------------------------------
# Export the accuracy results of all the models
# --------------------------------------------------------------

score_df.to_pickle("../../data/processed/score_df.pkl")
