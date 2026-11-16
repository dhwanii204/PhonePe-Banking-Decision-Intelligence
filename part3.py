
import csv
import numpy as np


# --------------------------------------------------
# TASK 3.1: K-MEANS CUSTOMER SEGMENTATION
# --------------------------------------------------

# Step 1: Load the dataset
with open("data/records.csv", "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    records = list(reader)

# Predictor columns X1 to X5
feature_columns = [
    "account_age_months",
    "failed_txn_count_90d",
    "credit_utilisation_ratio",
    "avg_monthly_inflow_inr_thousands",
    "repayment_score"
]

X = np.array(
    [
        [float(record[column]) for column in feature_columns]
        for record in records
    ],
    dtype=float
)

n = X.shape[0]

if n != 800:
    raise ValueError(f"Expected 800 records, found {n}")

print("Total records:", n)
print("Predictor shape:", X.shape)


# Step 2: Standardise X1 to X5
feature_means = np.mean(X, axis=0)
feature_sds = np.std(X, axis=0, ddof=1)

if np.any(feature_sds == 0):
    raise ValueError("A predictor has zero standard deviation.")

Z = (X - feature_means) / feature_sds

print("\nStandardisation completed.")
print("Standardised shape:", Z.shape)


# Step 3: Implement K-means from scratch
def kmeans(data, k):

    n_records = data.shape[0]

    # Initialise centroids using the specified row indices
    initial_indices = np.arange(k) * (n_records // k)

    centroids = data[initial_indices].copy()

    # Initially, no records have been assigned
    previous_labels = np.full(n_records, -1, dtype=int)

    passes = 0

    while True:

        # Calculate squared Euclidean distances
        differences = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]

        distances = np.sum(differences ** 2, axis=2)

        # np.argmin selects the lower index when distances tie
        labels = np.argmin(distances, axis=1)

        passes += 1

        # Check whether any customer changed clusters
        if np.array_equal(labels, previous_labels):
            break

        previous_labels = labels.copy()

        # Update each centroid using its cluster mean
        for cluster_index in range(k):

            cluster_points = data[labels == cluster_index]

            if len(cluster_points) > 0:
                centroids[cluster_index] = np.mean(
                    cluster_points,
                    axis=0
                )

            # Empty clusters retain their previous centroid

    # Calculate final WCSS
    final_differences = data - centroids[labels]

    wcss = np.sum(final_differences ** 2)

    return labels, centroids, wcss, passes


# Step 4: Run K-means for K = 2, 3, 4, 5, 6
kmeans_results = {}

print("\nK-means results")

for k in range(2, 7):

    labels, centroids, wcss, passes = kmeans(Z, k)

    kmeans_results[k] = {
        "labels": labels,
        "centroids": centroids,
        "wcss": wcss,
        "passes": passes
    }

    print(f"\nK = {k}")
    print(f"WCSS = {wcss:.4f}")
    print(f"Passes = {passes}")


# --------------------------------------------------
# TASK 3.1: ELBOW RULE AND CUSTOMER SEGMENTS
# --------------------------------------------------

# Step 5: Apply the elbow rule
elbow_k = None

print("\nElbow rule evaluation")

for k in range(3, 6):

    previous_reduction = (
        kmeans_results[k - 1]["wcss"]
        - kmeans_results[k]["wcss"]
    )

    next_reduction = (
        kmeans_results[k]["wcss"]
        - kmeans_results[k + 1]["wcss"]
    )

    condition = next_reduction < 0.5 * previous_reduction

    print(f"\nK = {k}")
    print(f"Previous reduction = {previous_reduction:.4f}")
    print(f"Next reduction = {next_reduction:.4f}")
    print("Elbow condition =", "yes" if condition else "no")

    if condition and elbow_k is None:
        elbow_k = k

print("\nSelected elbow K =", elbow_k)


# Step 6: Use K = 3 for customer segmentation
selected_k = 3

segment_labels = kmeans_results[selected_k]["labels"]


# Step 7: Order clusters by ascending original-unit mean X3
cluster_order = sorted(
    range(selected_k),
    key=lambda cluster: np.mean(
        X[segment_labels == cluster, 2]
    )
)

segment_names = ["A", "B", "C"]

# Read outcome and monthly spending
outcomes = np.array(
    [int(record["loan_default"]) for record in records]
)

monthly_spending = np.array(
    [
        float(record["monthly_spend_inr_thousands"])
        for record in records
    ]
)


# Step 8: Calculate segment profiles
segment_rows = []

print("\nCustomer segment profiles")

for segment_name, cluster_index in zip(
    segment_names, cluster_order
):

    mask = segment_labels == cluster_index

    cluster_X = X[mask]
    cluster_outcomes = outcomes[mask]
    cluster_spending = monthly_spending[mask]

    size = int(np.sum(mask))

    means = np.mean(cluster_X, axis=0)
    outcome_rate = np.mean(cluster_outcomes)
    mean_value = np.mean(cluster_spending)

    segment_rows.append([
        segment_name,
        size,
        *means,
        outcome_rate,
        mean_value
    ])

    print(f"\nSegment {segment_name}")
    print("Size =", size)

    for i, mean in enumerate(means, start=1):
        print(f"Mean X{i} = {mean:.4f}")

    print(f"Outcome rate = {outcome_rate:.4f}")
    print(f"Mean monthly spending = {mean_value:.4f}")


# Step 9: Generate the segments CSV
import os

os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/part3_segments.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "segment",
        "size",
        "mean_X1",
        "mean_X2",
        "mean_X3",
        "mean_X4",
        "mean_X5",
        "outcome_rate",
        "mean_value"
    ])

    for row in segment_rows:

        formatted_row = [
            row[0],
            row[1],
            *[f"{value:.4f}" for value in row[2:]]
        ]

        writer.writerow(formatted_row)


print("\nGenerated: outputs/part3_segments.csv")


# --------------------------------------------------
# TASK 3.2: HIERARCHICAL CLUSTERING
# --------------------------------------------------

# Step 1: Define the six-cohort dissimilarity matrix

cohort_names = ["E1", "E2", "E3", "E4", "E5", "E6"]

D = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])


# Step 2: Define the agglomerative clustering function

def hierarchical_clustering(distance_matrix, names, linkage):

    # Initially, every cohort is its own cluster
    clusters = [(name,) for name in names]

    merge_records = []

    merge_no = 0

    while len(clusters) > 1:

        best_distance = float("inf")
        best_pair = None

        # Examine every possible pair of clusters
        for i in range(len(clusters)):

            for j in range(i + 1, len(clusters)):

                cluster_a = clusters[i]
                cluster_b = clusters[j]

                pair_distances = [
                    distance_matrix[
                        names.index(member_a),
                        names.index(member_b)
                    ]
                    for member_a in cluster_a
                    for member_b in cluster_b
                ]

                # Calculate linkage distance
                if linkage == "single":
                    cluster_distance = min(pair_distances)

                elif linkage == "complete":
                    cluster_distance = max(pair_distances)

                elif linkage == "average":
                    cluster_distance = np.mean(pair_distances)

                else:
                    raise ValueError("Unknown linkage method")

                if cluster_distance < best_distance:
                    best_distance = cluster_distance
                    best_pair = (i, j)

        # Retrieve the selected clusters
        i, j = best_pair

        cluster_a = clusters[i]
        cluster_b = clusters[j]

        # Ensure cluster_a has the smaller minimum member
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a

        # Merge and sort cohort names alphabetically
        merged_cluster = tuple(
            sorted(cluster_a + cluster_b)
        )

        merge_no += 1

        # Record the merge
        merge_records.append({
            "linkage": linkage,
            "merge_no": merge_no,
            "cluster_a": "+".join(cluster_a),
            "cluster_b": "+".join(cluster_b),
            "height": best_distance
        })

        # Remove the old clusters
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (i, j)
        ]

        # Add the merged cluster
        clusters.append(merged_cluster)

        # Keep clusters in deterministic alphabetical order
        clusters.sort(key=lambda cluster: min(cluster))

    return merge_records


# Step 3: Run all three linkage methods

linkage_methods = ["single", "complete", "average"]

all_hierarchical_results = []

print("\nHierarchical clustering results")

for linkage in linkage_methods:

    results = hierarchical_clustering(
        D,
        cohort_names,
        linkage
    )

    all_hierarchical_results.extend(results)

    heights = np.array([
        record["height"]
        for record in results
    ])

    # Calculate consecutive merge-height gaps
    gaps = np.diff(heights)

    # Find the largest gap
    largest_gap_index = int(np.argmax(gaps))

    largest_gap = gaps[largest_gap_index]

    # Number of merges below the cut
    merges_below_cut = largest_gap_index + 1

    # Number of clusters remaining at the cut
    cluster_count = len(cohort_names) - merges_below_cut

    print(f"\nLinkage: {linkage}")

    for record in results:

        print(
            f"Merge {record['merge_no']}: "
            f"{record['cluster_a']} + "
            f"{record['cluster_b']} "
            f"at height {record['height']:.4f}"
        )

    print(f"Largest gap = {largest_gap:.4f}")
    print(f"Merges below cut = {merges_below_cut}")
    print(f"Clusters at cut = {cluster_count}")


# Step 4: Generate hierarchical clustering CSV

with open(
    "outputs/part3_hierarchical.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "linkage",
        "merge_no",
        "cluster_a",
        "cluster_b",
        "height"
    ])

    for record in all_hierarchical_results:

        writer.writerow([
            record["linkage"],
            record["merge_no"],
            record["cluster_a"],
            record["cluster_b"],
            f"{record['height']:.4f}"
        ])


print("\nGenerated: outputs/part3_hierarchical.csv")


# --------------------------------------------------
# TASK 3.3: PRINCIPAL COMPONENT ANALYSIS (PCA)
# --------------------------------------------------

# Step 1: Standardise the five predictors

X_pca = X.copy()

mean_pca = np.mean(X_pca, axis=0)
std_pca = np.std(X_pca, axis=0, ddof=1)

Z_pca = (X_pca - mean_pca) / std_pca

print("\nPCA results")

# Step 2: Calculate the covariance matrix

covariance_matrix = (
    Z_pca.T @ Z_pca
) / (Z_pca.shape[0] - 1)

# Step 3: Calculate eigenvalues and eigenvectors

eigenvalues, eigenvectors = np.linalg.eigh(
    covariance_matrix
)

# Sort eigenvalues in descending order

sort_indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[sort_indices]
eigenvectors = eigenvectors[:, sort_indices]

# Step 4: Apply consistent eigenvector signs
# The largest absolute loading in each component
# must be positive.

for column in range(eigenvectors.shape[1]):

    largest_loading_index = np.argmax(
        np.abs(eigenvectors[:, column])
    )

    if eigenvectors[largest_loading_index, column] < 0:
        eigenvectors[:, column] *= -1


# Step 5: Calculate explained variance

total_eigenvalue = np.sum(eigenvalues)

explained_variance_ratio = (
    eigenvalues / total_eigenvalue
)

cumulative_variance_ratio = np.cumsum(
    explained_variance_ratio
)

components_for_80 = int(
    np.argmax(cumulative_variance_ratio >= 0.80) + 1
)

print(f"Total eigenvalue sum = {total_eigenvalue:.4f}")

print("\nEigenvalues")

for i, value in enumerate(eigenvalues, start=1):
    print(f"PC{i} = {value:.4f}")

print("\nExplained variance ratios")

for i, value in enumerate(
    explained_variance_ratio,
    start=1
):
    print(f"PC{i} = {value:.4f}")

print("\nCumulative explained variance")

for i, value in enumerate(
    cumulative_variance_ratio,
    start=1
):
    print(f"PC{i} = {value:.4f}")

print(
    f"\nComponents needed for 80% variance = "
    f"{components_for_80}"
)


# Step 6: Display PC1 and PC2 loadings

predictor_names = ["X1", "X2", "X3", "X4", "X5"]

print("\nPC1 and PC2 loadings")

for i, name in enumerate(predictor_names):

    print(
        f"{name}: "
        f"PC1 = {eigenvectors[i, 0]:.4f}, "
        f"PC2 = {eigenvectors[i, 1]:.4f}"
    )


# Step 7: Project all records onto principal components

pca_scores = Z_pca @ eigenvectors

print("\nFirst three records: PCA projections")

for i in range(3):

    print(
        f"Record {i + 1}: "
        f"PC1 = {pca_scores[i, 0]:.4f}, "
        f"PC2 = {pca_scores[i, 1]:.4f}"
    )


# Step 8: Reproduce K-means K=3 labels
# This ensures PC1 averages use the same
# A/B/C segment definitions as Task 3.1.

k = 3

initial_indices = np.arange(k) * (
    Z_pca.shape[0] // k
)

centroids_pca = Z_pca[initial_indices].copy()

previous_labels = None

while True:

    distances = np.sum(
        (
            Z_pca[:, np.newaxis, :]
            - centroids_pca[np.newaxis, :, :]
        ) ** 2,
        axis=2
    )

    labels_pca = np.argmin(distances, axis=1)

    if (
        previous_labels is not None
        and np.array_equal(labels_pca, previous_labels)
    ):
        break

    previous_labels = labels_pca.copy()

    for cluster in range(k):

        members = Z_pca[labels_pca == cluster]

        if len(members) > 0:
            centroids_pca[cluster] = np.mean(
                members,
                axis=0
            )


# Sort clusters by ascending original-unit mean X3
# to reproduce the A/B/C labels.

cluster_order = sorted(
    range(k),
    key=lambda cluster: np.mean(
        X_pca[labels_pca == cluster, 2]
    )
)

segment_names = ["A", "B", "C"]

segment_mapping = {
    cluster: segment_names[position]
    for position, cluster in enumerate(cluster_order)
}

record_segments = np.array([
    segment_mapping[label]
    for label in labels_pca
])


# Step 9: Calculate mean PC1 score by segment

print("\nMean PC1 score by segment")

for segment in segment_names:

    segment_scores = pca_scores[
        record_segments == segment,
        0
    ]

    print(
        f"Segment {segment} = "
        f"{np.mean(segment_scores):.4f}"
    )

print("\nTask 3.3 PCA calculations completed.")



# --------------------------------------------------
# FINAL: GENERATE PART 3 RESULTS FILE
# --------------------------------------------------

results_lines = []

# ------------------------------
# TASK 3.1: K-MEANS RESULTS
# ------------------------------

results_lines.append("PART 3: CLUSTERING AND PCA")
results_lines.append("")
results_lines.append("TASK 3.1: K-MEANS")

results_lines.append(f"total_records={n}")
results_lines.append(f"selected_k={selected_k}")
results_lines.append(f"elbow_k={elbow_k}")

for k in range(2, 7):

    results_lines.append(
        f"wcss_k{k}={kmeans_results[k]['wcss']:.4f}"
    )

    results_lines.append(
        f"k{k}_passes={kmeans_results[k]['passes']}"
    )

for row in segment_rows:

    segment = row[0]

    results_lines.append(
        f"size_segment_{segment}={row[1]}"
    )

    for i in range(5):

        results_lines.append(
            f"segment_{segment}_mean_X{i + 1}={row[i + 2]:.4f}"
        )

    results_lines.append(
        f"segment_{segment}_outcome_rate={row[7]:.4f}"
    )

    results_lines.append(
        f"segment_{segment}_mean_value={row[8]:.4f}"
    )


# ------------------------------
# TASK 3.2: HIERARCHICAL RESULTS
# ------------------------------

results_lines.append("")
results_lines.append("TASK 3.2: HIERARCHICAL CLUSTERING")

for linkage in linkage_methods:

    linkage_records = [
        record
        for record in all_hierarchical_results
        if record["linkage"] == linkage
    ]

    heights = np.array([
        record["height"]
        for record in linkage_records
    ])

    gaps = np.diff(heights)

    largest_gap_index = int(np.argmax(gaps))

    largest_gap = gaps[largest_gap_index]

    merges_below_cut = largest_gap_index + 1

    cluster_count = len(cohort_names) - merges_below_cut

    results_lines.append("")
    results_lines.append(f"{linkage}_linkage")

    for record in linkage_records:

        results_lines.append(
            f"{linkage}_merge_{record['merge_no']}="
            f"{record['cluster_a']}+{record['cluster_b']},"
            f"{record['height']:.4f}"
        )

    results_lines.append(
        f"{linkage}_largest_gap={largest_gap:.4f}"
    )

    results_lines.append(
        f"{linkage}_merges_below_cut={merges_below_cut}"
    )

    results_lines.append(
        f"{linkage}_clusters_at_cut={cluster_count}"
    )


# ------------------------------
# TASK 3.3: PCA RESULTS
# ------------------------------

results_lines.append("")
results_lines.append("TASK 3.3: PRINCIPAL COMPONENT ANALYSIS")

results_lines.append(
    f"eigenvalue_sum={total_eigenvalue:.4f}"
)

for i in range(len(eigenvalues)):

    component_number = i + 1

    results_lines.append(
        f"eigenvalue_{component_number}={eigenvalues[i]:.4f}"
    )

    results_lines.append(
        f"explained_ratio_{component_number}="
        f"{explained_variance_ratio[i]:.4f}"
    )

    results_lines.append(
        f"cumulative_ratio_{component_number}="
        f"{cumulative_variance_ratio[i]:.4f}"
    )

results_lines.append(
    f"components_for_80pct={components_for_80}"
)

for i, name in enumerate(predictor_names):

    results_lines.append(
        f"pc1_loading_{name}={eigenvectors[i, 0]:.4f}"
    )

    results_lines.append(
        f"pc2_loading_{name}={eigenvectors[i, 1]:.4f}"
    )

for i in range(3):

    row_number = i + 1

    results_lines.append(
        f"row{row_number}_pc1={pca_scores[i, 0]:.4f}"
    )

    results_lines.append(
        f"row{row_number}_pc2={pca_scores[i, 1]:.4f}"
    )

for segment in segment_names:

    segment_scores = pca_scores[
        record_segments == segment,
        0
    ]

    results_lines.append(
        f"mean_pc1_segment_{segment}="
        f"{np.mean(segment_scores):.4f}"
    )


# ------------------------------
# WRITE FINAL RESULTS FILE
# ------------------------------

with open(
    "outputs/part3_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(results_lines) + "\n")

print("\nGenerated: outputs/part3_results.txt")
print("Part 3 results file created successfully.")