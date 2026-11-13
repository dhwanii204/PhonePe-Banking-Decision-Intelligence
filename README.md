Domain: Banking

## How to run

This project uses Python 3.10+ and NumPy 1.26.4. All datasets are synthetic and generated using `generate_data.py`.

To install the required dependency:

```bash
python -m pip install -r requirements.txt
```

To regenerate the synthetic dataset:

```bash
python generate_data.py
```

Run the analysis scripts from the project root directory:

```bash
python part1.py
```

The scripts calculate results and regenerate the corresponding output files inside the `outputs/` directory.

## Part 1 findings

### 1. Descriptive statistics and confidence interval

The analysis uses all 800 synthetic banking records. Average monthly customer spending is ₹8,938.20 thousand? No — the dataset measures spending in thousands of INR, so the mean is **₹8,938.20**.

The sample standard deviation is ₹9,172.30, and the standard error is ₹324.30. The 95% confidence interval for the population mean is approximately ₹8,302.60 to ₹9,573.80.

This interval provides a plausible range for the population's average monthly spending under the assumptions of the confidence interval calculation.

### 2. Hypothesis testing

The null hypothesis states that average monthly spending is ₹8,000, while the alternative hypothesis states that it differs from ₹8,000.

The calculated z-statistic is 2.8930, which exceeds the two-tailed critical value of 1.96 at the 5% significance level. Therefore, the null hypothesis is rejected.

A Type I error in this banking context would mean concluding that average monthly spending differs from ₹8,000 when the true population mean is actually ₹8,000.

The observed difference is ₹938.20. Although statistically significant, the bank should also consider whether this difference is practically meaningful for its customer analysis and business decisions.

### 3. Required sample size

The estimated minimum sample size required to achieve a margin of error of ₹500 at 95% confidence is 1,293 records.

The available dataset contains 800 records, which is 493 fewer than the estimated requirement. Therefore, a larger sample would be needed to meet the specified precision target.

### 4. Simple linear regression

A simple linear regression was used to examine the relationship between average monthly inflow and monthly spending.

The estimated slope is 0.6701. This indicates that an additional ₹1,000 in monthly inflow is associated with an estimated increase of approximately ₹670.10 in monthly spending.

The model has an R-squared value of 0.5196, meaning that approximately 51.96% of the variation in monthly spending is explained by monthly inflow in this linear model.

The RMSE is 6.3537 and the MAE is 3.7636, both measured in thousands of INR. The negative intercept should not be interpreted as realistic spending at zero inflow.

### 5. Gradient descent

Gradient descent was implemented using standardized monthly inflow, a learning rate of 0.1, and 500 full-batch updates.

The cost decreased from 163.9178 at iteration 0 to 40.3698 at iteration 500. The gradient-descent slope is 0.6701, and the slope gap compared with the closed-form regression is 0.000000.

This close agreement indicates that gradient descent converged to the same solution as the analytical simple linear regression for this dataset.

### 6. Residual analysis and log transformation

The residual standard deviation ratio for the raw linear model is 3.7525. The upper predicted-spending half has considerably greater residual variability than the lower half, suggesting a funnel-shaped pattern and non-constant residual variance.

After applying a log-log transformation, the residual standard deviation ratio decreases to 1.0396. This is much closer to 1, indicating more balanced residual variability between the two groups.

The log-log model has an R-squared value of 0.6565. Its RMSE is measured on the log-transformed spending scale, so it should not be directly compared with the raw model's RMSE.

### 7. Multiple linear regression

A multiple linear regression was fitted using log monthly spending as the dependent variable and seven predictors: log monthly inflow, account age, failed transaction count, credit utilisation, repayment score, irregular salary flag, and autopay flag.

Using the required absolute coefficient threshold of 0.05, the predictors that cross the threshold are:

- Log monthly inflow: 1.1576
- Failed transaction count: 0.0940
- Credit utilisation ratio: 0.1067
- Irregular salary flag: 0.1518

The multiple regression has an adjusted R-squared of 0.7403, compared with 0.6561 for the simple log-log regression. The multiple regression therefore explains more variation after accounting for the number of predictors.

The coefficients represent associations with log monthly spending while holding the other predictors constant. Their magnitudes should not be treated as directly comparable measures of feature importance because the predictors use different units and scales. The 0.05 threshold is a reporting rule, not a statistical significance test.