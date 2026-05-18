import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


st.title("Smartphone User Segmentation using K-Means")


# Upload CSV File
uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)


if uploaded_file is not None:

    # Load Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())


    # Select Numerical Columns
    X = df.select_dtypes(include=['int64', 'float64'])


    # Remove ID Columns
    for column in X.columns:
        if "id" in column.lower():
            X = X.drop(column, axis=1)


    # Feature Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    # Select Number of Clusters
    k = st.slider(
        "Select Number of Clusters",
        min_value=2,
        max_value=10,
        value=4
    )


    # Apply KMeans
    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    clusters = kmeans.fit_predict(X_scaled)


    # Add Cluster Labels
    df["Cluster"] = clusters


    # Silhouette Score
    score = silhouette_score(X_scaled, clusters)

    st.subheader("Silhouette Score")
    st.write(round(score, 3))


    # PCA for Visualization
    pca = PCA(n_components=2)

    pca_features = pca.fit_transform(X_scaled)


    # Create PCA DataFrame
    pca_df = pd.DataFrame(
        pca_features,
        columns=["PCA1", "PCA2"]
    )

    pca_df["Cluster"] = clusters


    # Plot Clusters
    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8,6))

    scatter = ax.scatter(
        pca_df["PCA1"],
        pca_df["PCA2"],
        c=pca_df["Cluster"]
    )

    ax.set_title("K-Means Clusters")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    st.pyplot(fig)