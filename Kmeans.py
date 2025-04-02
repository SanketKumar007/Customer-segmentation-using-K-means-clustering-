import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Streamlit App Title
st.title("Customer Segmentation using Unsupervised ML")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.write(df.head())

    # Store original categorical mappings
    category_mappings = {}

    # Data Preprocessing
    df.dropna(inplace=True)

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        category_mappings[col] = dict(enumerate(le.classes_))  # Store mappings

    # Standardization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Removing columns with more than 15 unique values
    selectable_columns = [col for col in df.columns if df[col].nunique() <= 15]

    # Histogram and Box Plot Notice
    st.write("### Histogram & Box Plot Selection")
    st.write("Only columns with 15 or fewer unique values are considered. Remaining columns are ignored.")

    # Histogram for Selected Feature
    st.write("### Histogram for Feature Distribution")
    selected_hist_col = st.selectbox("Select a column for Histogram", selectable_columns, index=0) if selectable_columns else None

    if selected_hist_col:
        plt.figure(figsize=(6, 4))
        if selected_hist_col in category_mappings:
            sns.histplot(df[selected_hist_col], bins=len(category_mappings[selected_hist_col]), kde=False)
            plt.xticks(ticks=list(category_mappings[selected_hist_col].keys()),
                       labels=list(category_mappings[selected_hist_col].values()), rotation=45)
        else:
            sns.histplot(df[selected_hist_col], kde=True, bins=30)
        plt.xlabel(selected_hist_col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {selected_hist_col}")
        st.pyplot(plt)
    else:
        st.write("Graph not possible")

    # Box Plot for Feature Distribution
    st.write("### Box Plot for Feature Distribution")
    selected_box_col = st.selectbox("Select a column for Box Plot", selectable_columns, index=0) if selectable_columns else None

    if selected_box_col:
        plt.figure(figsize=(6, 4))
        if selected_box_col in category_mappings:
            sns.boxplot(x=df[selected_box_col])
            plt.xticks(ticks=list(category_mappings[selected_box_col].keys()),
                       labels=list(category_mappings[selected_box_col].values()), rotation=45)
        else:
            sns.boxplot(x=df[selected_box_col])
        plt.xlabel(selected_box_col)
        plt.title(f"Box Plot of {selected_box_col}")
        st.pyplot(plt)
    else:
        st.write("Graph not possible")

    # Pairplot for Feature Relationships
    st.write("### Pairplot for Feature Relationships")
    selected_features = df.select_dtypes(include=['int64', 'float64']).columns[:4]
    if len(selected_features) >= 2:
        pairplot_fig = sns.pairplot(df[selected_features], diag_kind="kde")
        st.pyplot(pairplot_fig)
    else:
        st.write("Graph not possible")

    # Heatmap for Feature Correlation
    st.write("### Data Distribution via Heatmap")
    if df.shape[1] > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(pd.DataFrame(scaled_data).corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Index")
        st.pyplot(plt)
    else:
        st.write("Graph not possible")

    # Dimensionality Reduction with t-SNE
    st.write("### Visualizing High-Dimensional Data using t-SNE")
    if df.shape[1] > 1:
        tsne = TSNE(n_components=2, random_state=0)
        tsne_data = tsne.fit_transform(scaled_data)
        plt.figure(figsize=(7, 7))
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        st.pyplot(plt)
    else:
        st.write("Graph not possible")

    # Finding Optimal Clusters using Elbow Method
    st.write("### Finding Optimal Clusters using Elbow Method")
    if df.shape[0] > 1:
        errors = []
        for n_clusters in range(1, 11):
            model = KMeans(n_clusters=n_clusters, random_state=22)
            model.fit(scaled_data)
            errors.append(model.inertia_)
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=range(1, 11), y=errors, marker='o')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia (Sum of Squared Distances)")
        st.pyplot(plt)
    else:
        st.write("Graph not possible")

    # KMeans Clustering
    k = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)
    model = KMeans(n_clusters=k, random_state=22)
    df['Cluster'] = model.fit_predict(scaled_data)

    # Cluster Visualization
    st.write("### Cluster Visualization")
    if df.shape[0] > 1:
        df_tsne = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'Cluster': df['Cluster']})
        plt.figure(figsize=(7, 7))
        sns.scatterplot(x='x', y='y', hue='Cluster', palette='tab10', data=df_tsne)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        st.pyplot(plt)
    else:
        st.write("Graph not possible")

    # Clustered Data
    st.write("### Clustered Data (Categorized via K-Means Clustering)")
    header_row = pd.DataFrame([["Categorized via K-Means Clustering"] + [""] * (df.shape[1] - 1)], columns=df.columns)
    df_display = pd.concat([header_row, df], ignore_index=True)
    st.dataframe(df_display)
