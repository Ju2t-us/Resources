import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

def getCsvFiles():
    print("Loading data...")
    customer_data = pd.read_csv(r"Exit_Tickets\MarchSixth\travelers.csv", nrows=10000)
    print("Data loaded.")
    return customer_data

def encodeAndScaleData(df, numeric_columns, categorical_columns):
    print("Encoding and scaling data...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_columns),
            ("cat", OneHotEncoder(), categorical_columns),
        ]
    )
    processed_data = preprocessor.fit_transform(df)
    print("Data encoded and scaled.")
    return processed_data

def dbScan(processed_data, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(processed_data)
    return clustering.labels_

def find_optimal_eps(data, min_samples, eps_values):
    total = len(eps_values)
    print("Starting to find optimal eps...")
    best_eps = None
    best_score = -1
    for index, eps in enumerate(eps_values, start=1):
        print(f"Testing eps value {eps} ({index}/{total}, {index/total:.2%} complete)")
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = clustering.labels_
        if len(set(labels)) > 1 and np.count_nonzero(labels != -1) > 1:
            score = silhouette_score(data, labels)
            print(f"Eps: {eps} - Silhouette Score: {score}")
            if score > best_score:
                best_score = score
                best_eps = eps
    print(f"Optimal eps found: {best_eps} with score: {best_score}")
    return best_eps, best_score

def find_optimal_min_samples(data, best_eps, min_samples_values):
    total = len(min_samples_values)
    print("Starting to find optimal min_samples...")
    best_min_samples = None
    best_score = -1
    for index, min_samples in enumerate(min_samples_values, start=1):
        print(f"Testing min_samples value {min_samples} ({index}/{total}, {index/total:.2%} complete)")
        clustering = DBSCAN(eps=best_eps, min_samples=min_samples).fit(data)
        labels = clustering.labels_
        if len(set(labels)) > 1 and np.count_nonzero(labels != -1) > 1:
            score = silhouette_score(data, labels)
            print(f"Min_samples: {min_samples} - Silhouette Score: {score}")
            if score > best_score:
                best_score = score
                best_min_samples = min_samples
    print(f"Optimal min_samples found: {best_min_samples} with score: {best_score}")
    return best_min_samples, best_score

def plot_clusters(reduced_data, labels, clusters_to_plot=None):
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    if clusters_to_plot is not None:
        unique_labels = [x for x in unique_labels if x in clusters_to_plot]
    for label in unique_labels:
        color = "black" if label == -1 else plt.cm.jet(float(label) / max(unique_labels))
        plt.scatter(reduced_data[labels == label, 0], reduced_data[labels == label, 1], c=[color], label=f"Cluster {label}")
    plt.title("DBSCAN Clustering")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.legend()
    plt.show()

def reduce_dimensions(data, n_components=2):
    svd = TruncatedSVD(n_components=n_components)
    reduced_data = svd.fit_transform(data)
    return reduced_data


def plot_nearest_neighbors(data, n_neighbors=5):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)
    distances, indices = neigh.kneighbors(data)
    
    sorted_distances = np.sort(distances[:, n_neighbors-1])
    plt.figure(figsize=(10, 4))
    plt.plot(sorted_distances)
    plt.xlabel('Points')
    plt.ylabel('Distance')
    plt.title('Sorted Distances to {}-th Nearest Neighbor'.format(n_neighbors))
    plt.show()



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

    print("Processing customer data...")
    processed_customer_data = encodeAndScaleData(customer_data, numeric_columns, categorical_columns)

    # Plot nearest neighbors to help determine a good starting 'eps' value
    print("Plotting nearest neighbors...")
    plot_nearest_neighbors(processed_customer_data, n_neighbors=5)

    # Define the range for eps and min_samples values
    eps_values = np.linspace(1, 3, num=25)  # This will create 21 values between 2.0 and 2.5 inclusive
    min_samples_values = range(1, 20)  # Example range

    print("Finding optimal parameters for DBSCAN...")
    best_eps, eps_score = find_optimal_eps(processed_customer_data, 5, eps_values)
    print(f"Best Eps: {best_eps} with Silhouette Score: {eps_score}")

    if best_eps is not None:
        best_min_samples, min_samples_score = find_optimal_min_samples(
            processed_customer_data, best_eps, min_samples_values
        )
        print(f"Best Min_samples: {best_min_samples} with Silhouette Score: {min_samples_score}")

        if best_min_samples is not None:
            print("Running DBSCAN...")
            cluster_labels = dbScan(
                processed_customer_data, eps=best_eps, min_samples=best_min_samples
            )

            if len(cluster_labels) == len(customer_data):
                customer_data["Cluster"] = cluster_labels
                unique_clusters = np.unique(cluster_labels)
                num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
                print(f"Number of clusters: {num_clusters}")

                reduced_data = reduce_dimensions(processed_customer_data)
                plot_clusters(reduced_data, cluster_labels)
            else:
                print("Error: Mismatch in number of cluster labels and DataFrame rows.")
        else:
            print("No suitable min_samples value found.")
    else:
        print("No suitable eps value found.")

# Run the main function
main()
