import csv
import math
from pathlib import Path
import numpy as np

# Step 1: Read the dataset
data_path = Path("data") / "records.csv"

with open(data_path, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Step 2: Extract the monthly spending column
values = np.array(
    [float(row["monthly_spend_inr_thousands"]) for row in rows]
)
# Extract average monthly inflow (X4)
x4 = np.array(
    [float(row["avg_monthly_inflow_inr_thousands"]) for row in rows]
)

# Extract target and main predictor
values = np.array([float(row["monthly_spend_inr_thousands"]) for row in rows])
x4 = np.array([float(row["avg_monthly_inflow_inr_thousands"]) for row in rows])

# Extract remaining predictors for Task 1.7
x1 = np.array([float(row["account_age_months"]) for row in rows])
x2 = np.array([float(row["failed_txn_count_90d"]) for row in rows])
x3 = np.array([float(row["credit_utilisation_ratio"]) for row in rows])
x5 = np.array([float(row["repayment_score"]) for row in rows])
x6 = np.array([float(row["irregular_salary_flag"]) for row in rows])
x7 = np.array([float(row["autopay_flag"]) for row in rows])

# Step 3: Calculate descriptive statistics
n = len(values)
mean_value = np.mean(values)
sd_value = np.std(values, ddof=1)

# Step 4: Calculate standard error
se_value = sd_value / np.sqrt(n)

# Step 5: Calculate the 95% confidence interval
ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value

# Step 6: Hypothesis testing
hypothesised_mean = 8.0

z_stat = (mean_value - hypothesised_mean) / se_value

reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

# Step 7: Required sample size estimation

margin_of_error = 0.5

n_required = math.ceil(
    (1.96 * sd_value / margin_of_error) ** 2
)

# Step 8: Simple linear regression

# Calculate means
x_mean = np.mean(x4)
y_mean = np.mean(values)

# Calculate slope using covariance / variance
cov_xy = np.sum((x4 - x_mean) * (values - y_mean)) / (n - 1)
var_x = np.sum((x4 - x_mean) ** 2) / (n - 1)

reg_slope = cov_xy / var_x

# Calculate intercept
reg_intercept = y_mean - reg_slope * x_mean

# Calculate predicted spending
fitted = reg_intercept + reg_slope * x4

# Standardize X4 for gradient descent
x4_mean = np.mean(x4)
x4_sd = np.std(x4, ddof=1)

z = (x4 - x4_mean) / x4_sd

# Initialize gradient descent
m = 0.0
c = 0.0

learning_rate = 0.1
iterations = 500

gd_costs = {}

# Run gradient descent
for iteration in range(iterations + 1):

    # Calculate predictions
    predictions = m * z + c

    # Calculate errors
    errors = predictions - values

    # Calculate cost
    cost = np.mean(errors ** 2)

    # Store cost at required iterations
    if iteration in [0, 1, 10, 100, 500]:
        gd_costs[iteration] = cost

    # Update parameters (skip after final iteration)
    if iteration < iterations:

        dm = (2 / n) * np.sum(errors * z)
        dc = (2 / n) * np.sum(errors)

        m = m - learning_rate * dm
        c = c - learning_rate * dc

# Convert gradient descent coefficients to original X4 scale
gd_slope = m / x4_sd
gd_intercept = c - (m * x4_mean / x4_sd)

# Calculate difference between the two slopes
gd_slope_gap = abs(gd_slope - reg_slope)

# Task 1.6: Residual analysis

# Calculate residuals for the raw regression model
residuals_raw = values - fitted

# Sort customers by predicted spending (stable sorting)
raw_order = np.argsort(fitted, kind="stable")

# Divide into lower and upper halves
lower_raw = residuals_raw[raw_order[:400]]
upper_raw = residuals_raw[raw_order[400:]]

# Calculate sample standard deviations
resid_sd_lower_raw = np.std(lower_raw, ddof=1)
resid_sd_upper_raw = np.std(upper_raw, ddof=1)

# Calculate ratio
resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

# Log-log regression

log_x = np.log(x4)
log_y = np.log(values)

# Calculate means
log_x_mean = np.mean(log_x)
log_y_mean = np.mean(log_y)

# Calculate slope and intercept
log_cov = np.sum((log_x - log_x_mean) * (log_y - log_y_mean)) / (n - 1)
log_var = np.sum((log_x - log_x_mean) ** 2) / (n - 1)

loglog_slope = log_cov / log_var
loglog_intercept = log_y_mean - loglog_slope * log_x_mean

# Calculate fitted values and residuals
log_fitted = loglog_intercept + loglog_slope * log_x
log_residuals = log_y - log_fitted

# Calculate log-log model performance

log_tss = np.sum((log_y - log_y_mean) ** 2)
log_rss = np.sum((log_y - log_fitted) ** 2)

loglog_r2 = 1 - (log_rss / log_tss)
loglog_rmse = np.sqrt(log_rss / n)

# Sort log-log fitted values (stable sorting)
log_order = np.argsort(log_fitted, kind="stable")

# Divide log residuals into lower and upper halves
lower_log = log_residuals[log_order[:400]]
upper_log = log_residuals[log_order[400:]]

# Calculate sample standard deviations
resid_sd_lower_log = np.std(lower_log, ddof=1)
resid_sd_upper_log = np.std(upper_log, ddof=1)

# Calculate ratio
resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

# Task 1.7: Multiple Linear Regression

log_y = np.log(values)

design_matrix = np.column_stack((
    np.ones(n),
    np.log(x4),
    x1,
    x2,
    x3,
    x5,
    x6,
    x7
))

# Calculate multiple regression coefficients

beta = np.linalg.solve(
    design_matrix.T @ design_matrix,
    design_matrix.T @ log_y
)


# Calculate fitted values using the multiple regression model
mr_fitted = design_matrix @ beta

# Calculate total sum of squares and residual sum of squares
mr_tss = np.sum((log_y - np.mean(log_y)) ** 2)
mr_rss = np.sum((log_y - mr_fitted) ** 2)

# Calculate R-squared
mr_r2 = 1 - (mr_rss / mr_tss)

# Calculate adjusted R-squared for the multiple regression
p = 7
mr_adj_r2 = 1 - (1 - mr_r2) * (n - 1) / (n - p - 1)

# Calculate adjusted R-squared for the simple log-log regression
loglog_adj_r2 = 1 - (1 - loglog_r2) * (n - 1) / (n - 1 - 1)


# Calculate sum of squares
tss = np.sum((values - y_mean) ** 2)
ess = np.sum((fitted - y_mean) ** 2)
rss = np.sum((values - fitted) ** 2)

# Check the sum of squares identity
tss_equals_ess_plus_rss = (
    "yes" if np.isclose(tss, ess + rss) else "no"
)

# Calculate R-squared
r2 = ess / tss

# Calculate prediction errors
rmse = np.sqrt(rss / n)
mae = np.mean(np.abs(values - fitted))

# Step 9: Prepare the results
results = [
    f"n = {n}",
    f"mean_value = {mean_value:.4f}",
    f"sd_value = {sd_value:.4f}",
    f"se_value = {se_value:.4f}",
    f"ci95_low = {ci95_low:.4f}",
    f"ci95_high = {ci95_high:.4f}",
    f"z_stat = {z_stat:.4f}",
    f"reject_h0 = {reject_h0}",
    f"n_required = {n_required}",
    f"reg_slope = {reg_slope:.4f}",
    f"reg_intercept = {reg_intercept:.4f}",
    f"tss = {tss:.4f}",
    f"ess = {ess:.4f}",
    f"rss = {rss:.4f}",
    f"tss_equals_ess_plus_rss = {tss_equals_ess_plus_rss}",
    f"r2 = {r2:.4f}",
    f"rmse = {rmse:.4f}",
    f"mae = {mae:.4f}",
    f"gd_J0 = {gd_costs[0]:.4f}",
    f"gd_J1 = {gd_costs[1]:.4f}",
    f"gd_J10 = {gd_costs[10]:.4f}",
    f"gd_J100 = {gd_costs[100]:.4f}",
    f"gd_J500 = {gd_costs[500]:.4f}",
    f"gd_slope = {gd_slope:.4f}",
    f"gd_intercept = {gd_intercept:.4f}",
    f"gd_slope_gap = {gd_slope_gap:.4f}",
    f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}",
    f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}",
    f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}",
    f"loglog_intercept = {loglog_intercept:.4f}",
    f"loglog_slope = {loglog_slope:.4f}",
    f"loglog_r2 = {loglog_r2:.4f}",
    f"loglog_rmse = {loglog_rmse:.4f}",
    f"resid_sd_lower_log = {resid_sd_lower_log:.4f}",
    f"resid_sd_upper_log = {resid_sd_upper_log:.4f}",
    f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}",
    f"mr_beta_0 = {beta[0]:.4f}",
    f"mr_beta_1 = {beta[1]:.4f}",
    f"mr_beta_2 = {beta[2]:.4f}",
    f"mr_beta_3 = {beta[3]:.4f}",
    f"mr_beta_4 = {beta[4]:.4f}",
    f"mr_beta_5 = {beta[5]:.4f}",
    f"mr_beta_6 = {beta[6]:.4f}",
    f"mr_beta_7 = {beta[7]:.4f}",
    f"mr_r2 = {mr_r2:.4f}",
    f"mr_adj_r2 = {mr_adj_r2:.4f}",
    f"loglog_adj_r2 = {loglog_adj_r2:.4f}",
]

# Step 10: Save the results
output_path = Path("outputs") / "part1_results.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(results) + "\n", encoding="utf-8")

# Step 11: Display the results
print("\n".join(results))
