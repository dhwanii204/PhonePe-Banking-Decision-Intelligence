
import csv
import math
from pathlib import Path

import numpy as np


# --------------------------------------------------
# PART 2: CLASSIFICATION AND MODEL EVALUATION
# --------------------------------------------------

# Read the synthetic banking dataset
with open("data/records.csv", "r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

# Extract seven predictor variables
X = np.array([
    [
        float(row["account_age_months"]),
        float(row["failed_txn_count_90d"]),
        float(row["credit_utilisation_ratio"]),
        float(row["avg_monthly_inflow_inr_thousands"]),
        float(row["repayment_score"]),
        float(row["irregular_salary_flag"]),
        float(row["autopay_flag"])
    ]
    for row in rows
])

# Extract binary target variable
y = np.array([
    int(row["loan_default"])
    for row in rows
])

n = len(y)


# --------------------------------------------------
# TASK 2.1: STRATIFIED TRAIN-TEST SPLIT
# --------------------------------------------------

# Find the original row positions for each class
class_0_indices = np.where(y == 0)[0]
class_1_indices = np.where(y == 1)[0]

# Calculate 70% training positions for each class
class_0_train_count = (7 * len(class_0_indices)) // 10
class_1_train_count = (7 * len(class_1_indices)) // 10

# Select training and testing indices separately
train_indices = np.sort(np.concatenate((
    class_0_indices[:class_0_train_count],
    class_1_indices[:class_1_train_count]
)))

test_indices = np.sort(np.concatenate((
    class_0_indices[class_0_train_count:],
    class_1_indices[class_1_train_count:]
)))

# Create training and testing datasets
X_train_raw = X[train_indices]
X_test_raw = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

# Standardise using training data only
train_means = np.mean(X_train_raw, axis=0)
train_sds = np.std(X_train_raw, axis=0, ddof=1)

X_train = (X_train_raw - train_means) / train_sds
X_test = (X_test_raw - train_means) / train_sds


# Print split verification
print("Total records:", n)
print("Training records:", len(y_train))
print("Testing records:", len(y_test))
print("Training defaults:", int(np.sum(y_train)))
print("Testing defaults:", int(np.sum(y_test)))
print("Training predictor shape:", X_train.shape)
print("Testing predictor shape:", X_test.shape)


# --------------------------------------------------
# TASK 2.2: LOGISTIC REGRESSION FROM SCRATCH
# --------------------------------------------------

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# Binary cross-entropy loss
def binary_cross_entropy(y_true, probabilities):
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )

    return loss


# Initialise weights and bias
weights = np.zeros(X_train.shape[1])
bias = 0.0

learning_rate = 0.1
epochs = 1000

# Store losses at required epochs
loss_history = {}

# Training using full-batch gradient descent
for epoch in range(epochs + 1):

    # Calculate predicted probabilities
    train_probabilities = sigmoid(X_train @ weights + bias)

    # Calculate and store loss
    loss = binary_cross_entropy(y_train, train_probabilities)

    if epoch in [0, 1, 10, 100, 1000]:
        loss_history[epoch] = loss

    # Update weights and bias
    if epoch < epochs:
        error = train_probabilities - y_train

        weight_gradient = (X_train.T @ error) / len(y_train)
        bias_gradient = np.mean(error)

        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient


# Calculate final test probabilities
test_probabilities = sigmoid(X_test @ weights + bias)

# Calculate final test binary cross-entropy
test_bce = binary_cross_entropy(y_test, test_probabilities)

print("\nTask 2.2: Logistic Regression")
print("Initial loss:", f"{loss_history[0]:.4f}")
print("Loss after 1 epoch:", f"{loss_history[1]:.4f}")
print("Loss after 10 epochs:", f"{loss_history[10]:.4f}")
print("Loss after 100 epochs:", f"{loss_history[100]:.4f}")
print("Loss after 1000 epochs:", f"{loss_history[1000]:.4f}")
print("Test BCE:", f"{test_bce:.4f}")
print("Final bias:", f"{bias:.4f}")

for i, weight in enumerate(weights):
    print(f"Weight {i + 1}: {weight:.4f}")


# --------------------------------------------------
# TASK 2.3: LEARNING-RATE COMPARISON
# --------------------------------------------------

learning_rates = [0.01, 0.1, 1.0]
learning_rate_results = []

for lr in learning_rates:

    # Restart model parameters from zero
    trial_weights = np.zeros(X_train.shape[1])
    trial_bias = 0.0

    trial_losses = {}

    # Train for 200 full-batch updates
    for epoch in range(201):

        trial_probabilities = sigmoid(
            X_train @ trial_weights + trial_bias
        )

        trial_loss = binary_cross_entropy(
            y_train, trial_probabilities
        )

        if epoch in [0, 1, 10, 100, 200]:
            trial_losses[epoch] = trial_loss

        if epoch < 200:
            trial_error = trial_probabilities - y_train

            trial_weight_gradient = (
                X_train.T @ trial_error
            ) / len(y_train)

            trial_bias_gradient = np.mean(trial_error)

            trial_weights -= lr * trial_weight_gradient
            trial_bias -= lr * trial_bias_gradient

    learning_rate_results.append(
        (lr, trial_losses)
    )

    print(f"\nLearning rate: {lr}")
    for epoch in [0, 1, 10, 100, 200]:
        print(
            f"Loss at epoch {epoch}: "
            f"{trial_losses[epoch]:.4f}"
        )


# --------------------------------------------------
# TASK 2.4: ODDS RATIOS
# --------------------------------------------------

# Calculate odds ratios from the main logistic model
odds_ratios = np.exp(weights)

predictor_names = [
    "account_age_months",
    "failed_txn_count_90d",
    "credit_utilisation_ratio",
    "avg_monthly_inflow_inr_thousands",
    "repayment_score",
    "irregular_salary_flag",
    "autopay_flag"
]

print("\nTask 2.4: Odds Ratios")

for name, weight, odds_ratio in zip(
    predictor_names, weights, odds_ratios
):
    print(
        f"{name}: weight = {weight:.4f}, "
        f"odds_ratio = {odds_ratio:.4f}"
    )


# --------------------------------------------------
# TASK 2.5: CLASSIFICATION METRICS
# --------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):

    # Convert probabilities into predicted classes
    predictions = (probabilities >= threshold).astype(int)

    # Calculate confusion matrix values
    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    # Calculate evaluation metrics
    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }


print("\nTask 2.5: Classification Metrics")

for threshold in [0.5, 0.3]:

    metrics = calculate_metrics(
        y_test, test_probabilities, threshold
    )

    print(f"\nThreshold: {threshold}")

    for metric_name, metric_value in metrics.items():
        if metric_name in ["tp", "fp", "tn", "fn"]:
            print(f"{metric_name}: {metric_value}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")


# Calculate the accuracy of an all-negative baseline
blind_accuracy = np.mean(y_test == 0)

print("\nAll-negative baseline accuracy:", f"{blind_accuracy:.4f}")


# --------------------------------------------------
# TASK 2.6: THRESHOLD SWEEP
# --------------------------------------------------

# Create outputs directory if it does not exist
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

threshold_rows = []

for threshold in np.arange(0.1, 1.0, 0.1):

    metrics = calculate_metrics(
        y_test,
        test_probabilities,
        float(threshold)
    )

    tp = metrics["tp"]
    fp = metrics["fp"]
    tn = metrics["tn"]
    fn = metrics["fn"]

    # True positive rate and false positive rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    threshold_rows.append({
        "tau": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr
    })


# Write threshold results to CSV
threshold_file = output_dir / "part2_thresholds.csv"

with open(threshold_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "tau", "tp", "fp", "tn", "fn", "tpr", "fpr"
        ]
    )

    writer.writeheader()

    for row in threshold_rows:
        writer.writerow({
            "tau": f'{row["tau"]:.1f}',
            "tp": row["tp"],
            "fp": row["fp"],
            "tn": row["tn"],
            "fn": row["fn"],
            "tpr": f'{row["tpr"]:.4f}',
            "fpr": f'{row["fpr"]:.4f}'
        })


print("\nTask 2.6: Threshold Sweep")

for row in threshold_rows:
    print(
        f'tau={row["tau"]:.1f}, '
        f'tp={row["tp"]}, fp={row["fp"]}, '
        f'tn={row["tn"]}, fn={row["fn"]}, '
        f'tpr={row["tpr"]:.4f}, fpr={row["fpr"]:.4f}'
    )

print("Saved:", threshold_file)


# --------------------------------------------------
# TASK 2.7: K-NEAREST NEIGHBOURS
# --------------------------------------------------

def knn_predict(X_reference, y_reference, X_query, k):

    predictions = []
    probabilities = []

    for query_row in X_query:

        # Calculate Euclidean distance from this query
        # observation to every reference observation
        differences = X_reference - query_row
        distances = np.sqrt(np.sum(differences ** 2, axis=1))

        # Stable sorting preserves original file order
        # when two distances are equal
        nearest_indices = np.argsort(distances, kind="stable")[:k]

        # Get the outcomes of the K nearest observations
        nearest_outcomes = y_reference[nearest_indices]

        # Probability is the proportion of neighbours
        # whose outcome is default (1)
        probability = np.mean(nearest_outcomes)

        # Since K is odd, majority voting has no tie
        prediction = int(probability > 0.5)

        probabilities.append(probability)
        predictions.append(prediction)

    return (
        np.array(predictions, dtype=int),
        np.array(probabilities, dtype=float)
    )


def calculate_f1(y_true, predictions):

    tp = np.sum((y_true == 1) & (predictions == 1))
    fp = np.sum((y_true == 0) & (predictions == 1))
    fn = np.sum((y_true == 1) & (predictions == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# Five-fold cross-validation
fold_ids = np.arange(len(y_train)) % 5

k_values = range(1, 22, 2)
knn_cv_rows = []

print("\nTask 2.7: KNN Five-Fold Cross-Validation")

for k in k_values:

    fold_f1_scores = []

    for fold in range(5):

        # Current fold is used for validation
        validation_indices = np.where(fold_ids == fold)[0]

        # Remaining four folds are used as reference data
        reference_indices = np.where(fold_ids != fold)[0]

        X_reference = X_train[reference_indices]
        y_reference = y_train[reference_indices]

        X_validation = X_train[validation_indices]
        y_validation = y_train[validation_indices]

        # Predict the validation fold
        validation_predictions, _ = knn_predict(
            X_reference,
            y_reference,
            X_validation,
            k
        )

        # Calculate F1 for this fold
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

    print(f"K={k}, mean CV F1={mean_cv_f1:.4f}")


# Select the K with the highest mean F1.
# Smaller K wins if scores are tied.
best_knn_row = max(
    knn_cv_rows,
    key=lambda row: (row["mean_cv_f1"], -row["K"])
)

best_k = best_knn_row["K"]

print("\nSelected K:", best_k)
print("Best mean CV F1:", f'{best_knn_row["mean_cv_f1"]:.4f}')


# Save cross-validation results
knn_cv_file = output_dir / "part2_knn_cv.csv"

with open(knn_cv_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["K", "mean_cv_f1"]
    )

    writer.writeheader()

    for row in knn_cv_rows:
        writer.writerow({
            "K": row["K"],
            "mean_cv_f1": f'{row["mean_cv_f1"]:.4f}'
        })


# Refit using all training observations and evaluate on test data
knn_test_predictions, knn_test_probabilities = knn_predict(
    X_train,
    y_train,
    X_test,
    best_k
)

knn_test_metrics = calculate_metrics(
    y_test,
    knn_test_probabilities,
    0.5
)

print("\nKNN Test Metrics")
print("Selected K:", best_k)

for metric_name, metric_value in knn_test_metrics.items():

    if metric_name in ["tp", "fp", "tn", "fn"]:
        print(f"{metric_name}: {metric_value}")
    else:
        print(f"{metric_name}: {metric_value:.4f}")


print("Saved:", knn_cv_file)


# --------------------------------------------------
# TASK 2.8: THRESHOLD SELECTION USING INR COST
# --------------------------------------------------

# Recalculate probabilities using the final Task 2.2
# logistic regression model.
# This ensures we use the logistic model, not KNN.

logistic_train_probabilities = sigmoid(
    X_train @ weights + bias
)

logistic_test_probabilities = sigmoid(
    X_test @ weights + bias
)


# Evaluate thresholds from 0.05 to 0.95
cost_rows = []

for k in range(1, 20):

    threshold = k / 20

    # Evaluate this threshold on TRAINING data
    train_metrics = calculate_metrics(
        y_train,
        logistic_train_probabilities,
        threshold
    )

    # Calculate training cost
    train_cost = (
        50 * train_metrics["fp"]
        + 400 * train_metrics["fn"]
    )

    cost_rows.append({
        "tau": threshold,
        "cost_train": train_cost
    })


# Select lowest training cost.
# If costs are tied, select the smaller threshold.
best_cost_row = min(
    cost_rows,
    key=lambda row: (row["cost_train"], row["tau"])
)

cost_tau = best_cost_row["tau"]
cost_train_at_tau = best_cost_row["cost_train"]


# Evaluate the selected threshold on TEST data
test_metrics_at_tau = calculate_metrics(
    y_test,
    logistic_test_probabilities,
    cost_tau
)

cost_test_at_tau = (
    50 * test_metrics_at_tau["fp"]
    + 400 * test_metrics_at_tau["fn"]
)


# Calculate test cost at the standard threshold of 0.5
test_metrics_at_05 = calculate_metrics(
    y_test,
    logistic_test_probabilities,
    0.5
)

cost_test_at_05 = (
    50 * test_metrics_at_05["fp"]
    + 400 * test_metrics_at_05["fn"]
)


# Flag nobody: all actual defaults become missed positives
cost_test_flag_nobody = 400 * int(np.sum(y_test == 1))


print("\nTask 2.8: Threshold Selection Using INR Cost")

print("\nTraining threshold costs:")

for row in cost_rows:
    print(
        f'Threshold: {row["tau"]:.2f}, '
        f'Training cost: INR {row["cost_train"]}'
    )

print("\nSelected threshold:", f"{cost_tau:.2f}")
print("Training cost at selected threshold:", cost_train_at_tau)

print("\nTest cost at selected threshold:", cost_test_at_tau)
print("Test cost at threshold 0.5:", cost_test_at_05)
print("Test cost when flagging nobody:", cost_test_flag_nobody)


# --------------------------------------------------
# SAVE PART 2 RESULTS
# --------------------------------------------------

# Calculate logistic regression metrics at both required thresholds
logistic_metrics_05 = calculate_metrics(
    y_test, logistic_test_probabilities, 0.5
)

logistic_metrics_03 = calculate_metrics(
    y_test, logistic_test_probabilities, 0.3
)


# Collect results using the exact key names from the project brief
results = {
    # Task 2.1: Dataset split
    "total_records": n,
    "train_rows": len(y_train),
    "train_positives": int(np.sum(y_train)),
    "test_rows": len(y_test),
    "test_positives": int(np.sum(y_test)),

    # Task 2.2: Logistic regression
    "loss_epoch_0": loss_history[0],
    "loss_epoch_1": loss_history[1],
    "loss_epoch_10": loss_history[10],
    "loss_epoch_100": loss_history[100],
    "loss_epoch_1000": loss_history[1000],
    "test_bce": test_bce,
    "bias": bias,

    # Logistic regression weights
    "weight_X1": weights[0],
    "weight_X2": weights[1],
    "weight_X3": weights[2],
    "weight_X4": weights[3],
    "weight_X5": weights[4],
    "weight_X6": weights[5],
    "weight_X7": weights[6],

    # Task 2.3: Learning-rate comparison (200 updates)
    "loss_epoch_200_eta_0.01": learning_rate_results[0][1][200],
    "loss_epoch_200_eta_0.1": learning_rate_results[1][1][200],
    "loss_epoch_200_eta_1.0": learning_rate_results[2][1][200],

    # Task 2.4: Odds ratios
    "odds_ratio_X1": odds_ratios[0],
    "odds_ratio_X2": odds_ratios[1],
    "odds_ratio_X3": odds_ratios[2],
    "odds_ratio_X4": odds_ratios[3],
    "odds_ratio_X5": odds_ratios[4],
    "odds_ratio_X6": odds_ratios[5],
    "odds_ratio_X7": odds_ratios[6],

    # Task 2.5: Logistic metrics at threshold 0.5
    "tau_0.5_tp": logistic_metrics_05["tp"],
    "tau_0.5_fp": logistic_metrics_05["fp"],
    "tau_0.5_tn": logistic_metrics_05["tn"],
    "tau_0.5_fn": logistic_metrics_05["fn"],
    "tau_0.5_accuracy": logistic_metrics_05["accuracy"],
    "tau_0.5_precision": logistic_metrics_05["precision"],
    "tau_0.5_recall": logistic_metrics_05["recall"],
    "tau_0.5_f1": logistic_metrics_05["f1"],
    "tau_0.5_specificity": logistic_metrics_05["specificity"],

    # Task 2.5: Logistic metrics at threshold 0.3
    "tau_0.3_tp": logistic_metrics_03["tp"],
    "tau_0.3_fp": logistic_metrics_03["fp"],
    "tau_0.3_tn": logistic_metrics_03["tn"],
    "tau_0.3_fn": logistic_metrics_03["fn"],
    "tau_0.3_accuracy": logistic_metrics_03["accuracy"],
    "tau_0.3_precision": logistic_metrics_03["precision"],
    "tau_0.3_recall": logistic_metrics_03["recall"],
    "tau_0.3_f1": logistic_metrics_03["f1"],
    "tau_0.3_specificity": logistic_metrics_03["specificity"],

    "blind_accuracy": blind_accuracy,

    # Task 2.7: KNN
    "knn_best_k": best_k,
    "knn_best_cv_f1": best_knn_row["mean_cv_f1"],
    "knn_tp": knn_test_metrics["tp"],
    "knn_fp": knn_test_metrics["fp"],
    "knn_tn": knn_test_metrics["tn"],
    "knn_fn": knn_test_metrics["fn"],
    "knn_accuracy": knn_test_metrics["accuracy"],
    "knn_precision": knn_test_metrics["precision"],
    "knn_recall": knn_test_metrics["recall"],
    "knn_f1": knn_test_metrics["f1"],
    "knn_specificity": knn_test_metrics["specificity"],

    # Task 2.8: INR cost comparison
    "cost_tau": cost_tau,
    "cost_train_at_tau": cost_train_at_tau,
    "cost_test_at_tau": cost_test_at_tau,
    "cost_test_at_0.5": cost_test_at_05,
    "cost_test_flag_nobody": cost_test_flag_nobody
}

# Write results to a text file
results_file = output_dir / "part2_results.txt"

with open(results_file, "w", encoding="utf-8") as file:
    for key, value in results.items():

        if isinstance(value, (int, np.integer)):
            formatted_value = str(int(value))
        else:
            formatted_value = f"{float(value):.4f}"

        file.write(f"{key} = {formatted_value}\n")

print("\nPart 2 results saved:", results_file)
