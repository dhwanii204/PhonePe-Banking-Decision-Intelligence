
import csv
import numpy as np


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

with open(
    "data/records.csv",
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)
    records = list(reader)


# Original predictors X1 to X7
X = np.array(
    [
        [
            float(record["account_age_months"]),
            float(record["failed_txn_count_90d"]),
            float(record["credit_utilisation_ratio"]),
            float(record["avg_monthly_inflow_inr_thousands"]),
            float(record["repayment_score"]),
            float(record["irregular_salary_flag"]),
            float(record["autopay_flag"])
        ]
        for record in records
    ],
    dtype=float
)

# Binary target: loan default
y = np.array(
    [int(record["loan_default"]) for record in records],
    dtype=int
)


# --------------------------------------------------
# RECREATE THE EXACT PART 2 TRAIN-TEST SPLIT
# --------------------------------------------------

class_0_indices = np.where(y == 0)[0]
class_1_indices = np.where(y == 1)[0]

class_0_train_count = (7 * len(class_0_indices)) // 10
class_1_train_count = (7 * len(class_1_indices)) // 10

train_indices = np.sort(
    np.concatenate((
        class_0_indices[:class_0_train_count],
        class_1_indices[:class_1_train_count]
    ))
)

test_indices = np.sort(
    np.concatenate((
        class_0_indices[class_0_train_count:],
        class_1_indices[class_1_train_count:]
    ))
)

X_train_raw = X[train_indices]
y_train = y[train_indices]

print("Total records:", len(records))
print("Training records:", len(y_train))
print("Testing records:", len(test_indices))
print("Training defaults:", int(np.sum(y_train)))
print("Training non-defaults:", int(len(y_train) - np.sum(y_train)))


# --------------------------------------------------
# TASK 4.1: DECISION-TREE SPLIT CRITERIA
# --------------------------------------------------

def calculate_entropy(target):

    if len(target) == 0:
        return 0.0

    positive_rate = np.mean(target)
    negative_rate = 1 - positive_rate

    entropy = 0.0

    if positive_rate > 0:
        entropy -= positive_rate * np.log2(positive_rate)

    if negative_rate > 0:
        entropy -= negative_rate * np.log2(negative_rate)

    return float(entropy)


def calculate_gini(target):

    if len(target) == 0:
        return 0.0

    positive_rate = np.mean(target)
    negative_rate = 1 - positive_rate

    return float(
        1 - positive_rate ** 2 - negative_rate ** 2
    )


# Parent-node impurity
parent_entropy = calculate_entropy(y_train)
parent_gini = calculate_gini(y_train)
root_entropy = parent_entropy
root_gini = parent_gini

print("\nParent node")
print("Entropy =", f"{parent_entropy:.4f}")
print("Gini =", f"{parent_gini:.4f}")


# Four candidate split rules
split_rules = [
    ("S1", "X6", X_train_raw[:, 5] == 1),
    ("S2", "X7", X_train_raw[:, 6] == 1),
    ("S3", "X3", X_train_raw[:, 2] > 0.5),
    ("S4", "X1", X_train_raw[:, 0] <= 24)
]


split_rows = []

print("\nCandidate split results")

for split_name, predictor_name, yes_mask in split_rules:

    # Divide target values into YES and NO groups
    yes_target = y_train[yes_mask]
    no_target = y_train[~yes_mask]

    # Group sizes
    n_yes = len(yes_target)
    n_no = len(no_target)

    # Default counts
    positives_yes = int(np.sum(yes_target))
    positives_no = int(np.sum(no_target))

    # Weighted child entropy
    weighted_entropy = (
        (n_yes / len(y_train)) * calculate_entropy(yes_target)
        + (n_no / len(y_train)) * calculate_entropy(no_target)
    )

    # Weighted child Gini
    weighted_gini = (
        (n_yes / len(y_train)) * calculate_gini(yes_target)
        + (n_no / len(y_train)) * calculate_gini(no_target)
    )

    # Information gain and Gini gain
    information_gain = parent_entropy - weighted_entropy
    gini_gain = parent_gini - weighted_gini

    split_rows.append({
        "split": split_name,
        "n_yes": n_yes,
        "positives_yes": positives_yes,
        "n_no": n_no,
        "positives_no": positives_no,
        "info_gain": information_gain,
        "gini_gain": gini_gain
    })

    print("\n", split_name, "-", predictor_name)
    print("YES group size =", n_yes)
    print("YES group defaults =", positives_yes)
    print("NO group size =", n_no)
    print("NO group defaults =", positives_no)
    print("Information gain =", f"{information_gain:.4f}")
    print("Gini gain =", f"{gini_gain:.4f}")


# --------------------------------------------------
# SELECT BEST SPLIT BY INFORMATION GAIN
# --------------------------------------------------

best_split = max(
    split_rows,
    key=lambda row: row["info_gain"]
)
best_split_by_info_gain = best_split["split"]

# Best split according to each criterion
best_split_by_info_gain = best_split["split"]

best_split_by_gini_gain = max(
    split_rows,
    key=lambda row: row["gini_gain"]
)["split"]

# Stump probabilities for the best split (X1 <= 24)
stump_yes_mask = X_train_raw[:, 0] <= 24

stump_rate_yes = float(np.mean(y_train[stump_yes_mask]))
stump_rate_no = float(np.mean(y_train[~stump_yes_mask]))

# Expected out-of-bag fraction for a bootstrap sample
oob_fraction = (1 - 1 / len(y_train)) ** len(y_train)

print("\nBest split by information gain:", best_split["split"])
print("Best information gain =", f'{best_split["info_gain"]:.4f}')


# --------------------------------------------------
# WRITE TASK 4.1 CSV
# --------------------------------------------------

with open(
    "outputs/part4_splits.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "split",
        "n_yes",
        "positives_yes",
        "n_no",
        "positives_no",
        "info_gain",
        "gini_gain"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in split_rows:

        writer.writerow({
            "split": row["split"],
            "n_yes": row["n_yes"],
            "positives_yes": row["positives_yes"],
            "n_no": row["n_no"],
            "positives_no": row["positives_no"],
            "info_gain": f'{row["info_gain"]:.4f}',
            "gini_gain": f'{row["gini_gain"]:.4f}'
        })


print("\nGenerated: outputs/part4_splits.csv")
print("Task 4.1 completed.")


# --------------------------------------------------
# TASK 4.2: ENSEMBLE VOTING
# --------------------------------------------------

# Standardise using training data only,
# matching Part 2 exactly.

train_means = np.mean(X_train_raw, axis=0)
train_sds = np.std(X_train_raw, axis=0, ddof=1)

X_train = (X_train_raw - train_means) / train_sds

X_test_raw = X[test_indices]
X_test = (X_test_raw - train_means) / train_sds

y_test = y[test_indices]


# --------------------------------------------------
# LOGISTIC REGRESSION
# --------------------------------------------------

def sigmoid(values):

    values = np.clip(values, -500, 500)

    return 1 / (1 + np.exp(-values))


weights = np.zeros(X_train.shape[1])
bias = 0.0

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):

    train_probabilities = sigmoid(
        X_train @ weights + bias
    )

    error = train_probabilities - y_train

    weight_gradient = (
        X_train.T @ error
    ) / len(y_train)

    bias_gradient = np.mean(error)

    weights -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient


logistic_test_probabilities = sigmoid(
    X_test @ weights + bias
)


# --------------------------------------------------
# KNN FUNCTIONS
# --------------------------------------------------

def knn_predict(X_reference, y_reference, X_query, k):

    predictions = []
    probabilities = []

    for query_row in X_query:

        differences = X_reference - query_row

        distances = np.sqrt(
            np.sum(differences ** 2, axis=1)
        )

        nearest_indices = np.argsort(
            distances,
            kind="stable"
        )[:k]

        nearest_outcomes = y_reference[nearest_indices]

        probability = np.mean(nearest_outcomes)

        # Match Part 2's strict majority rule
        prediction = int(probability > 0.5)

        predictions.append(prediction)
        probabilities.append(probability)

    return (
        np.array(predictions, dtype=int),
        np.array(probabilities, dtype=float)
    )


def calculate_f1(y_true, predictions):

    tp = np.sum((y_true == 1) & (predictions == 1))
    fp = np.sum((y_true == 0) & (predictions == 1))
    fn = np.sum((y_true == 1) & (predictions == 0))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------
# FIVE-FOLD CROSS-VALIDATION
# --------------------------------------------------

fold_ids = np.arange(len(y_train)) % 5

knn_cv_rows = []

for k in range(1, 22, 2):

    fold_f1_scores = []

    for fold in range(5):

        validation_indices = np.where(
            fold_ids == fold
        )[0]

        reference_indices = np.where(
            fold_ids != fold
        )[0]

        X_reference = X_train[reference_indices]
        y_reference = y_train[reference_indices]

        X_validation = X_train[validation_indices]
        y_validation = y_train[validation_indices]

        validation_predictions, _ = knn_predict(
            X_reference,
            y_reference,
            X_validation,
            k
        )

        fold_f1 = calculate_f1(
            y_validation,
            validation_predictions
        )

        fold_f1_scores.append(fold_f1)

    mean_cv_f1 = float(np.mean(fold_f1_scores))

    knn_cv_rows.append({
        "K": k,
        "mean_cv_f1": mean_cv_f1
    })


best_knn_row = max(
    knn_cv_rows,
    key=lambda row: (
        row["mean_cv_f1"],
        -row["K"]
    )
)

best_k = best_knn_row["K"]


# Final KNN probabilities using all training data
knn_test_predictions_05, knn_test_probabilities = knn_predict(
    X_train,
    y_train,
    X_test,
    best_k
)


# --------------------------------------------------
# DECISION STUMP
# --------------------------------------------------

# Use S4, the best information-gain split:
# YES: account age <= 24 months
# NO: account age > 24 months

stump_yes_probability = (
    best_split["positives_yes"]
    / best_split["n_yes"]
)

stump_no_probability = (
    best_split["positives_no"]
    / best_split["n_no"]
)

stump_test_probabilities = np.where(
    X_test_raw[:, 0] <= 24,
    stump_yes_probability,
    stump_no_probability
)


# --------------------------------------------------
# INDIVIDUAL MODEL PREDICTIONS AT THRESHOLD 0.3
# --------------------------------------------------

threshold = 0.3

logistic_predictions = (
    logistic_test_probabilities >= threshold
).astype(int)

knn_predictions = knn_test_predictions_05.copy()

stump_predictions = (
    stump_test_probabilities >= threshold
).astype(int)


# --------------------------------------------------
# HARD VOTING
# --------------------------------------------------

all_predictions = np.column_stack((
    logistic_predictions,
    knn_predictions,
    stump_predictions
))

hard_vote_predictions = (
    np.sum(all_predictions, axis=1) >= 2
).astype(int)


# --------------------------------------------------
# SOFT VOTING
# --------------------------------------------------

all_probabilities = np.column_stack((
    logistic_test_probabilities,
    knn_test_probabilities,
    stump_test_probabilities
))

soft_vote_probabilities = np.mean(
    all_probabilities,
    axis=1
)

soft_vote_predictions = (
    soft_vote_probabilities >= threshold
).astype(int)


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def evaluate_predictions(actual, predicted):

    tp = int(np.sum(
        (actual == 1) & (predicted == 1)
    ))

    fp = int(np.sum(
        (actual == 0) & (predicted == 1)
    ))

    tn = int(np.sum(
        (actual == 0) & (predicted == 0)
    ))

    fn = int(np.sum(
        (actual == 1) & (predicted == 0)
    ))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


voting_models = [
    ("logistic", logistic_predictions),
    ("knn", knn_predictions),
    ("stump", stump_predictions),
    ("hard_vote", hard_vote_predictions),
    ("soft_vote", soft_vote_predictions)
]

voting_rows = []

print("\nTask 4.2: Ensemble voting")
print("Selected K =", best_k)
print("Best CV F1 =", f'{best_knn_row["mean_cv_f1"]:.4f}')

for model_name, predictions in voting_models:

    metrics = evaluate_predictions(
        y_test,
        predictions
    )

    row = {
        "model": model_name,
        **metrics
    }

    voting_rows.append(row)

    print("\nModel:", model_name)

    for key, value in metrics.items():

        if isinstance(value, float):
            print(key, "=", f"{value:.4f}")
        else:
            print(key, "=", value)


# --------------------------------------------------
# WRITE VOTING CSV
# --------------------------------------------------

with open(
    "outputs/part4_voting.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "model",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in voting_rows:

        writer.writerow({
            "model": row["model"],
            "tp": row["tp"],
            "fp": row["fp"],
            "tn": row["tn"],
            "fn": row["fn"],
            "precision": f'{row["precision"]:.4f}',
            "recall": f'{row["recall"]:.4f}',
            "f1": f'{row["f1"]:.4f}'
        })


print("\nGenerated: outputs/part4_voting.csv")
print("Task 4.2 completed.")


# --------------------------------------------------
# TASK 4.3: ADABOOST WEIGHT UPDATES
# --------------------------------------------------

# Actual labels
# +1 = positive class
# -1 = negative class

y_ada = np.array([
    1, 1, 1, -1, -1, -1, 1, -1
])

# Predictions from weak classifier 1
h1 = np.array([
    1, 1, -1, -1, -1, -1, -1, -1
])

# Predictions from weak classifier 2
h2 = np.array([
    -1, 1, 1, -1, 1, -1, 1, -1
])


# Initially, every observation has equal weight
initial_weights = np.full(
    len(y_ada),
    1 / len(y_ada)
)

adaboost_rows = []

print("\nTask 4.3: AdaBoost")


# --------------------------------------------------
# ROUND 1: WEAK CLASSIFIER 1
# --------------------------------------------------

weights_before_round1 = initial_weights.copy()

# Weighted classification error
error1 = np.sum(
    weights_before_round1 * (y_ada != h1)
)

# Classifier importance
alpha1 = 0.5 * np.log(
    (1 - error1) / error1
)

# Update observation weights
unnormalised1 = (
    weights_before_round1
    * np.exp(-alpha1 * y_ada * h1)
)

# Normalise so that weights sum to 1
weights_after_round1 = (
    unnormalised1 / np.sum(unnormalised1)
)

print("\nRound 1")
print("Weighted error =", f"{error1:.4f}")
print("Alpha 1 =", f"{alpha1:.4f}")


for i in range(len(y_ada)):

    adaboost_rows.append({
        "round": 1,
        "point": i + 1,
        "weight_before": weights_before_round1[i],
        "weight_unnormalised": unnormalised1[i],
        "weight_after": weights_after_round1[i]
    })


# --------------------------------------------------
# ROUND 2: WEAK CLASSIFIER 2
# --------------------------------------------------

# Round 2 starts with the normalised Round 1 weights
weights_before_round2 = weights_after_round1.copy()

error2 = np.sum(
    weights_before_round2 * (y_ada != h2)
)

alpha2 = 0.5 * np.log(
    (1 - error2) / error2
)

unnormalised2 = (
    weights_before_round2
    * np.exp(-alpha2 * y_ada * h2)
)

weights_after_round2 = (
    unnormalised2 / np.sum(unnormalised2)
)

print("\nRound 2")
print("Weighted error =", f"{error2:.4f}")
print("Alpha 2 =", f"{alpha2:.4f}")


for i in range(len(y_ada)):

    adaboost_rows.append({
        "round": 2,
        "point": i + 1,
        "weight_before": weights_before_round2[i],
        "weight_unnormalised": unnormalised2[i],
        "weight_after": weights_after_round2[i]
    })


# --------------------------------------------------
# FINAL ENSEMBLE PREDICTION
# --------------------------------------------------

ensemble_scores = (
    alpha1 * h1
    + alpha2 * h2
)

ensemble_predictions = np.where(
    ensemble_scores >= 0,
    1,
    -1
)

ensemble_accuracy = np.mean(
    ensemble_predictions == y_ada
)

print("\nFinal AdaBoost ensemble")
print("Ensemble accuracy =", f"{ensemble_accuracy:.4f}")

print("\nPoint-wise ensemble results")

for i in range(len(y_ada)):

    print(
        "Point", i + 1,
        "| Actual =", y_ada[i],
        "| Predicted =", ensemble_predictions[i],
        "| Score =", f"{ensemble_scores[i]:.4f}"
    )


# --------------------------------------------------
# WRITE ADABOOST CSV
# --------------------------------------------------

with open(
    "outputs/part4_adaboost.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "round",
        "point",
        "weight_before",
        "weight_unnormalised",
        "weight_after"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in adaboost_rows:

        writer.writerow({
            "round": row["round"],
            "point": row["point"],
            "weight_before": f'{row["weight_before"]:.4f}',
            "weight_unnormalised": f'{row["weight_unnormalised"]:.4f}',
            "weight_after": f'{row["weight_after"]:.4f}'
        })


print("\nGenerated: outputs/part4_adaboost.csv")
print("Task 4.3 completed.")


# --------------------------------------------------
# TASK 4.4: NEURAL NETWORK ARCHITECTURE PARAMETERS
# --------------------------------------------------

architecture_rows = []


def add_architecture_row(layer, parameters):

    architecture_rows.append({
        "layer": layer,
        "parameters": int(parameters)
    })


# --------------------------------------------------
# A. MULTI-LAYER PERCEPTRON (MLP)
# --------------------------------------------------

# Input: 784
# Hidden layer 1: 16
# Hidden layer 2: 16
# Output: 10

mlp_hidden1 = (784 * 16) + 16
mlp_hidden2 = (16 * 16) + 16
mlp_output = (16 * 10) + 10

mlp_total = (
    mlp_hidden1
    + mlp_hidden2
    + mlp_output
)

add_architecture_row("mlp_hidden1", mlp_hidden1)
add_architecture_row("mlp_hidden2", mlp_hidden2)
add_architecture_row("mlp_output", mlp_output)


# --------------------------------------------------
# B. CONVOLUTIONAL NEURAL NETWORK (CNN)
# --------------------------------------------------

# Input: 28 x 28 x 1

# Conv1:
# Kernel = 5 x 5
# Input channels = 1
# Filters = 6
# Stride = 1
# Padding = 2

cnn_conv1 = (
    (5 * 5 * 1) + 1
) * 6

# Conv1 output: 28 x 28 x 6
# MaxPool output: 14 x 14 x 6

# Conv2:
# Kernel = 5 x 5
# Input channels = 6
# Filters = 16
# Stride = 1
# Padding = 0

cnn_conv2 = (
    (5 * 5 * 6) + 1
) * 16

# Conv2 output: 10 x 10 x 16
# MaxPool output: 5 x 5 x 16

# Flatten: 5 x 5 x 16 = 400

cnn_dense1 = (400 * 32) + 32

cnn_dense2 = (32 * 10) + 10

cnn_total = (
    cnn_conv1
    + cnn_conv2
    + cnn_dense1
    + cnn_dense2
)

add_architecture_row("cnn_conv1", cnn_conv1)
add_architecture_row("cnn_conv2", cnn_conv2)
add_architecture_row("cnn_dense1", cnn_dense1)
add_architecture_row("cnn_dense2", cnn_dense2)



# --------------------------------------------------
# C. FULLY CONNECTED LAYER
# --------------------------------------------------

# Input image: 128 x 128 x 3
# Output neurons: 1024

fc_input_features = 128 * 128 * 3

fc_parameters = (
    fc_input_features * 1024
) + 1024




# --------------------------------------------------
# D. CONVOLUTIONAL LAYER
# --------------------------------------------------

# Input: 128 x 128 x 3
# Kernel: 3 x 3
# Filters: 64
# One bias per filter

conv_parameters = (
    (3 * 3 * 3) + 1
) * 64




# --------------------------------------------------
# E. RECURRENT NEURAL NETWORK (RNN)
# --------------------------------------------------

# Input size = 64
# Hidden size = 128
# Output size = 16

W_xh = 64 * 128
W_hh = 128 * 128
W_hy = 128 * 16

b_h = 128
b_y = 16

rnn_total = (
    W_xh
    + W_hh
    + W_hy
    + b_h
    + b_y
)



# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\nTask 4.4: Architecture parameters")

for row in architecture_rows:

    print(
        row["layer"],
        "=",
        row["parameters"]
    )


# --------------------------------------------------
# PRINT CNN OUTPUT SHAPES
# --------------------------------------------------

print("\nCNN output shapes")

print("Input:", "28 x 28 x 1")
print("Conv1:", "28 x 28 x 6")
print("MaxPool1:", "14 x 14 x 6")
print("Conv2:", "10 x 10 x 16")
print("MaxPool2:", "5 x 5 x 16")
print("Flatten:", "400")
print("Dense1:", "32")
print("Output:", "10")


# --------------------------------------------------
# PRINT RNN PARAMETER BREAKDOWN
# --------------------------------------------------

print("\nRNN parameter breakdown")

print("W_xh =", W_xh)
print("W_hh =", W_hh)
print("W_hy =", W_hy)
print("b_h =", b_h)
print("b_y =", b_y)
print("RNN total =", rnn_total)

print(
    "RNN parameters remain unchanged "
    "when sequence length changes."
)


# --------------------------------------------------
# WRITE ARCHITECTURE CSV
# --------------------------------------------------

with open(
    "outputs/part4_architecture.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["layer", "parameters"]
    )

    writer.writeheader()
    writer.writerows(architecture_rows)


print("\nGenerated: outputs/part4_architecture.csv")
print("Task 4.4 completed.")


# --------------------------------------------------
# WRITE PART 4 TEXT RESULTS
# --------------------------------------------------

result_lines = []

# TASK 4.1: DECISION-TREE SPLIT CRITERIA

result_lines.append("Task 4.1: Decision-Tree Split Criteria")
result_lines.append(f"training_records = {len(y_train)}")
result_lines.append(f"training_defaults = {int(np.sum(y_train))}")
result_lines.append(
    f"training_non_defaults = {int(len(y_train) - np.sum(y_train))}"
)
result_lines.append(f"parent_entropy = {parent_entropy:.4f}")
result_lines.append(f"parent_gini = {parent_gini:.4f}")
result_lines.append(f"root_entropy = {root_entropy:.4f}")
result_lines.append(f"root_gini = {root_gini:.4f}")
result_lines.append(f"best_split = {best_split['split']}")
result_lines.append(
    f"best_information_gain = {best_split['info_gain']:.4f}"
)
result_lines.append(
    f"best_split_by_info_gain = {best_split_by_info_gain}"
)
result_lines.append(
    f"best_split_by_gini_gain = {best_split_by_gini_gain}"
)
result_lines.append(f"stump_rate_yes = {stump_rate_yes:.4f}")
result_lines.append(f"stump_rate_no = {stump_rate_no:.4f}")
result_lines.append(f"oob_fraction = {oob_fraction:.4f}")

for row in split_rows:
    result_lines.append(
        f"{row['split']}_information_gain = {row['info_gain']:.4f}"
    )
    result_lines.append(
        f"{row['split']}_gini_gain = {row['gini_gain']:.4f}"
    )

# TASK 4.2: ENSEMBLE VOTING

result_lines.append("")
result_lines.append("Task 4.2: Ensemble Voting")
result_lines.append(f"selected_knn_k = {best_k}")
result_lines.append(
    f"knn_best_cv_f1 = {best_knn_row['mean_cv_f1']:.4f}"
)

for row in voting_rows:
    model = row["model"]

    result_lines.append(f"{model}_tp = {row['tp']}")
    result_lines.append(f"{model}_fp = {row['fp']}")
    result_lines.append(f"{model}_tn = {row['tn']}")
    result_lines.append(f"{model}_fn = {row['fn']}")
    result_lines.append(f"{model}_precision = {row['precision']:.4f}")
    result_lines.append(f"{model}_recall = {row['recall']:.4f}")
    result_lines.append(f"{model}_f1 = {row['f1']:.4f}")

# TASK 4.3: ADABOOST

result_lines.append("")
result_lines.append("Task 4.3: AdaBoost")

result_lines.append(f"round1_error = {error1:.4f}")
result_lines.append(f"round1_alpha = {alpha1:.4f}")
result_lines.append(f"round2_error = {error2:.4f}")
result_lines.append(f"round2_alpha = {alpha2:.4f}")
result_lines.append(
    f"ensemble_accuracy = {ensemble_accuracy:.4f}"
)

round1_misclassified_weight = np.sum(
    weights_after_round1[y_ada != h1]
)
result_lines.append(
    f"round1_misclassified_weight = {round1_misclassified_weight:.4f}"
)

for i in range(len(y_ada)):
    result_lines.append(
        f"point_{i + 1}_actual = {int(y_ada[i])}"
    )
    result_lines.append(
        f"point_{i + 1}_predicted = {int(ensemble_predictions[i])}"
    )
    result_lines.append(
        f"point_{i + 1}_round1_weight = "
        f"{weights_after_round1[i]:.4f}"
    )

# TASK 4.4: NEURAL NETWORK ARCHITECTURES

result_lines.append("")
result_lines.append("Task 4.4: Neural Network Architectures")

result_lines.append(f"mlp_total = {mlp_total}")
result_lines.append(f"cnn_total = {cnn_total}")
result_lines.append(f"fc_layer_params = {fc_parameters}")
result_lines.append(f"conv_layer_params = {conv_parameters}")
result_lines.append(f"rnn_total = {rnn_total}")

# Include the seven individual MLP and CNN layer counts
with open(
    "outputs/part4_architecture.csv",
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        result_lines.append(
            f"{row['layer']}_parameters = {int(row['parameters'])}"
        )

# Write the complete results file
with open(
    "outputs/part4_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(result_lines) + "\n")

print("\nGenerated: outputs/part4_results.txt")
print("Part 4 results file completed.")






