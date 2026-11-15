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
python part2.py
python part3.py
python part4.py
```

The scripts calculate results and regenerate the corresponding output files inside the `outputs/` directory.

## Part 1 findings

### 1. Descriptive statistics and confidence interval

The analysis uses all 800 synthetic banking records. Mean monthly customer spending is 8.9382 thousand INR (₹8,938.20).

The sample standard deviation is 9.1723 thousand INR (₹9,172.30), and the standard error is 0.3243 thousand INR (₹324.30). The 95% confidence interval for the population mean is approximately ₹8,302.60 to ₹9,573.80.

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

## Part 2 findings

### 1. Stratified train-test split and standardisation

The 800 synthetic banking records were divided into 560 training records and 240 testing records using a stratified 70:30 split.

The training set contains 119 loan defaults, while the testing set contains 51. This preserves the default-class proportions across the two datasets.

The seven predictor variables were standardised using the training-set means and sample standard deviations. The same training statistics were then applied to the test set to avoid using test information during preprocessing.

### 2. Logistic regression

A logistic regression classifier was implemented from scratch using the sigmoid function, binary cross-entropy loss, and full-batch gradient descent.

With a learning rate of 0.1 and 1,000 updates, training loss decreased from 0.6931 initially to 0.3704. The test binary cross-entropy was 0.3818, indicating that the model also achieved a similar level of predictive fit on unseen test records.

### 3. Learning-rate comparison

Three learning rates were compared over 200 full-batch updates, with the model parameters restarted from zero for each trial.

- Learning rate 0.01: final loss 0.5091.
- Learning rate 0.1: final loss 0.3776.
- Learning rate 1.0: final loss 0.3703.

A learning rate of 1.0 achieved the lowest loss after 200 updates in this experiment. The 0.01 rate converged more slowly. The main logistic regression model nevertheless retains the prescribed learning rate of 0.1 and 1,000 updates.

### 4. Odds-ratio interpretation

The odds ratios were calculated by exponentiating the logistic regression coefficients. Because the predictors were standardised, each continuous predictor's odds ratio describes the change in default odds associated with a one-standard-deviation increase, holding the other predictors constant.

Credit utilisation has the largest positive coefficient (0.9013) and an odds ratio of 2.4629. Failed transaction count also has a positive association with default odds, with an odds ratio of 1.7341. The irregular salary flag has an odds ratio of 1.2646.

Account age, average monthly inflow, repayment score, and autopay have negative coefficients in this fitted model. These are model associations, not evidence of causal effects or guaranteed individual outcomes.

### 5. Classification performance and threshold selection

At the 0.5 threshold, logistic regression achieved 80.42% accuracy, 0.5455 precision, 0.4706 recall, and an F1-score of 0.5053.

At the 0.3 threshold, accuracy decreased to 76.25%, while recall increased to 0.7451 and F1-score increased to 0.5714. This lower threshold identifies more actual defaults but also produces more false positives.

The all-negative baseline achieves 78.75% accuracy. Therefore, accuracy alone does not fully describe the model's usefulness, particularly when identifying defaults is important.

### 6. Threshold sweep

The test-set threshold sweep shows the trade-off between true positive rate and false positive rate.

At a threshold of 0.1, the model identifies 48 of 51 defaults, giving a true positive rate of 0.9412, but the false positive rate is 0.4656.

At a threshold of 0.8, the false positive rate falls to zero, but only 2 defaults are identified. This demonstrates why the classification threshold should reflect the relative importance of missed defaults and false alerts.

### 7. K-nearest neighbours

A K-nearest neighbours classifier was implemented using Euclidean distance on the standardised predictors. Five-fold cross-validation was used to compare odd values of K from 1 to 21.

K=5 achieved the highest mean cross-validation F1-score of 0.4541 and was selected for testing.

On the test set, KNN achieved an F1-score of 0.3696 and recall of 0.3333. This is lower than logistic regression at both the 0.5 and 0.3 thresholds. For this dataset and evaluation setup, logistic regression provides stronger default-detection performance.

### 8. Cost-sensitive threshold selection

The cost-sensitive analysis assigns ₹50 to each false positive and ₹400 to each false negative. Thresholds from 0.05 to 0.95 were evaluated using training-set logistic probabilities.

The selected training threshold was 0.15, with a training cost of ₹12,150. Applying this threshold to the test set resulted in a cost of ₹6,050.

For comparison, the test cost at the standard threshold of 0.5 was ₹11,800, while flagging nobody would have resulted in a cost of ₹20,400 because all 51 actual defaults would be missed.

The threshold of 0.15 was selected using training data only. Its test cost is reported as an evaluation result, not used to choose the threshold. Under the specified cost assumptions, this threshold is preferable to 0.5 for the evaluated test set. These synthetic results should not be treated as a real lending policy or credit decision rule.


## Part 3 findings

### 1. K-means customer segmentation

K-means clustering was implemented from scratch using the five standardised predictors: account age, failed transaction count, credit utilisation, average monthly inflow, and repayment score.

The algorithm was evaluated for K values from 2 to 6. The WCSS decreased from 2,262.6753 at K=2 to 1,278.1691 at K=6. The specified elbow rule selected K=3, which was also used for the final customer segmentation.

The resulting segments contain 322 customers in Segment A, 255 in Segment B, and 223 in Segment C.

Segment A has the lowest average credit utilisation (0.2250), the highest average repayment score (62.1584), and the lowest observed default rate (1.86%). This segment shows comparatively favourable repayment characteristics in the synthetic dataset.

Segment B has the highest average monthly inflow (28.7192 thousand INR) and the highest average monthly spending (17.1248 thousand INR). Its observed default rate is 23.53%.

Segment C has the highest average credit utilisation (0.7118), the lowest average repayment score (38.5215), and the highest observed default rate (46.64%). This segment shows comparatively higher default incidence in the synthetic dataset.

These profiles can help a bank understand differences between customer groups and design further analysis or targeted financial support. Cluster membership alone should not be used to approve or reject individual customers.

### 2. Hierarchical clustering

Agglomerative hierarchical clustering was implemented using single, complete, and average linkage on the supplied six-cohort dissimilarity matrix.

All three methods initially merge E3 and E6. They subsequently add E2 to that group and combine E4 with E5, followed by E1.

The largest consecutive merge-height gap was 1.4000 for single linkage, 5.0000 for complete linkage, and 2.7556 for average linkage.

Using the specified largest-gap cutting rule, single linkage produces four clusters, whereas complete and average linkage each produce two clusters.

The difference occurs because single linkage uses the minimum pairwise distance, complete linkage uses the maximum pairwise distance, and average linkage uses the mean pairwise distance between clusters.

The results demonstrate how the choice of linkage method can influence the final grouping, even when all methods use the same dissimilarity matrix.

### 3. Principal component analysis

Principal component analysis was performed on the five standardised predictors using the covariance matrix and eigenvalue decomposition.

The sum of the eigenvalues is 5.0000, matching the number of standardised predictors.

The first principal component explains 52.27% of the total variance, while the second explains 23.66%. Together, they explain 75.93% of the variance.

The first three principal components explain 86.16% of the total variance. Therefore, three components are required to retain at least 80% of the variation.

For PC1, account age has a loading of 0.5317 and repayment score has a loading of 0.4418. Credit utilisation has a loading of -0.5187, failed transaction count has a loading of -0.4100, and average monthly inflow has a loading of -0.2916.

This component represents a contrast between account age and repayment score on one side and credit utilisation and failed transactions on the other. These loadings describe statistical relationships and should not be interpreted as causal effects.

The mean PC1 scores are 1.7773 for Segment A, -0.7831 for Segment B, and -1.6709 for Segment C. These results provide another view of the differences between the customer segments identified through K-means.

PCA reduces the dimensionality of the predictor data, but the components may be less directly interpretable than the original banking variables.

## Part 4 findings

### 1. Decision-tree split criteria

The root node has an entropy of 0.7462 and a Gini impurity of 0.3347. Split S4 (X1 ≤ 24) is the best split because it produces the highest information gain of 0.1252 and the highest Gini gain of 0.0606 among the four candidates. Growing a tree until every leaf is pure can cause overfitting because it may learn noise and individual patterns instead of general relationships; setting a maximum tree depth or minimum leaf size can help prevent this.

The expected out-of-bag fraction for 560 bootstrap draws is 0.3676. The stump's estimated default probabilities are 0.4574 for the YES group and 0.0887 for the NO group.

### 2. Ensemble voting

Among the five evaluated models, logistic regression is the strongest choice based on F1-score. At the specified 0.3 threshold, it achieves an F1-score of 0.5714 and recall of 0.7451.

Under the common 0.3 probability threshold used in this comparison, the decision stump achieves an F1-score of 0.5532 and recall of 0.7647, while KNN achieves an F1-score of 0.4806 and recall of 0.6078. Hard voting produces an F1-score of 0.5481, and soft voting produces 0.5507. Neither voting method improves on logistic regression's F1-score, so combining these models does not provide a performance advantage in this experiment.

### 3. AdaBoost

The first weak learner has a weighted error of 0.2500 and an alpha of 0.5493. After its weight update and normalisation, the two misclassified points (points 3 and 7) each receive a weight of 0.2500, giving a combined weight of 0.5000.

The second weak learner has a lower weighted error of 0.1667 and a higher alpha of 0.8047. Therefore, AdaBoost gives h2 more influence than h1 because h2 performs better on the updated, more challenging observations. The final ensemble accuracy is 0.7500.

### 4. Neural network architecture sizing

#### Model A: CNN for cheque-digit recognition

The CNN option for recognising handwritten digits has the following output shapes and parameter counts:

| Layer | Output shape | Parameters |
|---|---:|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv1 (5 × 5, 6 filters) | 28 × 28 × 6 | 156 |
| ReLU | 28 × 28 × 6 | 0 |
| MaxPool1 (2 × 2) | 14 × 14 × 6 | 0 |
| Conv2 (5 × 5, 16 filters) | 10 × 10 × 16 | 2,416 |
| ReLU | 10 × 10 × 16 | 0 |
| MaxPool2 (2 × 2) | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense1 (32 neurons) | 32 | 12,832 |
| Output (10 neurons) | 10 | 330 |

The CNN has 15,734 trainable parameters in total. The alternative fully connected MLP has 13,002 parameters.

For comparison, a fully connected layer mapping a 128 × 128 × 3 input to 1,024 neurons requires 50,332,672 parameters, whereas a convolutional layer with 64 filters of size 3 × 3 requires only 1,792 parameters. Convolution uses shared filter weights, which greatly reduces the number of parameters.

For a 10-class classification task, ReLU is a suitable hidden-layer activation, while softmax at the output with categorical cross-entropy is a suitable combination for class probabilities.

#### Model B: RNN for payment-remark classification

For the RNN, the input-to-hidden weight matrix W_xh has shape 128 × 64, the hidden-to-hidden matrix W_hh has shape 128 × 128, and the hidden-to-output matrix W_hy has shape 16 × 128.

Including the biases, the RNN contains 26,768 parameters. The same weights are reused at every time step, so increasing the sequence length increases computation but does not increase the number of trainable parameters.

A very deep plain network can suffer from vanishing gradients, where gradients become progressively smaller as they pass backwards through many layers, preventing early layers from learning effectively. Skip connections provide shorter paths for information and gradients to travel, helping preserve gradient flow during training.

## Model-choice memo

**To:** Head of the Lending Risk Team
**Subject:** Structured-data model recommendation

I recommend logistic regression as the structured-data model to take forward for controlled production validation. At the 0.3 threshold, it achieves an F1-score of 0.5714 and recall of 0.7451, outperforming hard voting (F1 0.5481) and soft voting (F1 0.5507). Its recall is also higher than KNN's 0.6078.

For the cost-based operating threshold, I recommend 0.15, selected using training data. On the test set, logistic regression at this threshold incurs a cost of INR 6,050, compared with INR 11,800 at the standard 0.5 threshold and INR 20,400 when no borrowers are flagged. These costs use INR 50 per false positive and INR 400 per false negative.

The evidence has limitations: it comes from 800 synthetic records and one stratified train-test split, so it does not establish real-world lending performance. The threshold and model should be validated on representative, real-world data before deployment.

Model B, the payment-remark classification RNN, is worth a limited pilot before production because payment remarks may provide additional lending-risk context; its incremental value must be tested rather than assumed.

## Limitations

- **Synthetic data:** All 800 records are generated artificially and do not represent actual PhonePe or partner-lender customers. The results cannot establish real-world model performance.
- **Single train-test split:** Classification results use one stratified 70:30 split. Performance may vary with a different sample or split.
- **Limited positive test records:** The test set contains only 51 defaults, so a small number of changed predictions can noticeably affect precision, recall and F1-score.
- **Simplified cost assumptions:** The cost analysis uses fixed costs of INR 50 per false positive and INR 400 per false negative. Actual collection and recovery costs may differ.
- **No deployment validation:** The models have not been tested for fairness, data drift, operational reliability, or performance on new borrowers.
- **Architecture sizing only:** The MLP, CNN and RNN architectures were sized mathematically but were not trained or evaluated on actual cheque images or payment remarks.