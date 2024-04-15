import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture

def getCsvFiles():
    print("Loading data...")
    customer_data = pd.read_csv(r"Exit_Tickets\MarchSixth\travelers.csv", nrows=1000)
    print("Data loaded.")
    return customer_data

def encodeAndScaleData(df, numeric_columns, categorical_columns):
    print("Encoding and scaling data...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_columns),
            ("cat", OneHotEncoder(sparse_output=False), categorical_columns),
        ]
    )
    processed_data = preprocessor.fit_transform(df)
    print("Data encoded and scaled.")
    return processed_data

def gmmCluster(processed_data, n_components):
    gmm = GaussianMixture(n_components=n_components).fit(processed_data)
    labels = gmm.predict(processed_data)
    return labels

def find_optimal_components(data, component_values):
    total = len(component_values)
    print("Starting to find optimal number of components...")
    best_n_components = None
    best_score = -1
    for index, n_components in enumerate(component_values, start=1):
        print(f"Testing number of components {n_components} ({index}/{total}, {index/total:.2%} complete)")
        labels = gmmCluster(data, n_components)
        if len(set(labels)) > 1:
            score = silhouette_score(data, labels)
            print(f"Number of components: {n_components} - Silhouette Score: {score}")
            if score > best_score:
                best_score = score
                best_n_components = n_components
    print(f"Optimal number of components found: {best_n_components} with score: {best_score}")
    return best_n_components, best_score

def plot_clusters(reduced_data, labels, clusters_to_plot=None):
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    if clusters_to_plot is not None:
        unique_labels = [x for x in unique_labels if x in clusters_to_plot]
    for label in unique_labels:
        color = plt.cm.jet(float(label) / max(unique_labels))
        plt.scatter(reduced_data[labels == label, 0], reduced_data[labels == label, 1], c=[color], label=f"Cluster {label}")
    plt.title("GMM Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

def reduce_dimensions(data, n_components=2):
    svd = TruncatedSVD(n_components=n_components)
    reduced_data = svd.fit_transform(data)
    return reduced_data

def describe_clusters(data, labels, numeric_columns, categorical_columns):
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        cluster_data = data[(labels == cluster)]
        print(f"Cluster {cluster}:")
        for col in numeric_columns:
            print(f"  Average {col}: {cluster_data[col].mean()}")
        for col in categorical_columns:
            print(f"  Most common {col}: {cluster_data[col].mode()[0]}")
        print()

def main():
    print("Starting main function...")
    customer_data = getCsvFiles()

    numeric_columns = [
        "Age", 
        "Distance to Destination (Light-Years)", 
        "Duration of Stay (Earth Days)", 
        "Number of Companions", 
        "Price (Galactic Credits)", 
        "Customer Satisfaction Score"
    ]

    categorical_columns = [
        "Gender", 
        "Occupation", 
        "Travel Class", 
        "Destination", 
        "Star System", 
        "Purpose of Travel", 
        "Transportation Type", 
        "Loyalty Program Member", 
        "Month"
    ]

    scenarios = [
        {"name": "Financial-Focused", "numeric": ["Price (Galactic Credits)"], "categorical": ["Loyalty Program Member"]},
    ]

    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        processed_customer_data = encodeAndScaleData(customer_data, scenario["numeric"], scenario["categorical"])
        component_values = range(1, 21)
        print("Finding optimal parameters for GMM...")
        best_n_components, component_score = find_optimal_components(processed_customer_data, component_values)
        print(f"Best number of components: {best_n_components} with Silhouette Score: {component_score}")

        if best_n_components is not None and component_score > 0.9:
            print("Running GMM...")
            cluster_labels = gmmCluster(processed_customer_data, best_n_components)
            if len(cluster_labels) == len(customer_data):
                reduced_data = reduce_dimensions(processed_customer_data)
                plot_clusters(reduced_data, cluster_labels)
                describe_clusters(customer_data, cluster_labels, scenario["numeric"], scenario["categorical"])
            else:
                print("Error: Mismatch in number of cluster labels and DataFrame rows.")
        else:
            print("No suitable model found with a high enough silhouette score.")

# Run the main function
main()
