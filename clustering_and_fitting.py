"""
Clustering And Fitting.py

This script performs data cleaning, visualization, clustering, and polynomial
fitting on the Spotify dataset.
It has been updated to match the lecturer’s GitHub template.
"""

# Importing essential libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_squared_error
from scipy.stats import skew, kurtosis, t
from numpy.polynomial.polynomial import Polynomial


# Data Loading and Cleaning Functions
def load_and_clean_data(data):
    """
    Load and clean the dataset from a CSV file.

    Parameters:
        filename (str): The path to the CSV file to load.

    Returns:
        pd.DataFrame: A cleaned DataFrame with irrelevant columns removed and
        missing values dropped.
    """
    df = pd.read_csv(data)

    # Inspecting data (optional prints – can be commented out in production)
    print("Head of data:")
    print(df.head(), "\n")
    print("Tail of data:")
    print(df.tail(), "\n")
    print("Summary statistics:")
    print(df.describe(), "\n")
    print("Data info:")
    print(df.info(), "\n")
    print("Missing values per column:")
    print(df.isnull().sum(), "\n")
    print("Duplicate rows count:", df.duplicated().sum(), "\n")

    # Drop irrelevant columns (check if they exist)
    columns_to_drop = ["uri", "track_href", "analysis_url", "id", "type"]
    existing_columns = df.columns
    columns_to_drop = [
        col for col in columns_to_drop if col in existing_columns]
    df.drop(columns=columns_to_drop, inplace=True)

    # Remove any rows with missing values
    df.dropna(inplace=True)

    return df


# Visualization Functions
def relational_plot(dataframe, x_var, y_var, color_var=None):
    """
    Create a relational scatter plot for two variables, optionally colored by
    a third variable.

    Parameters:
        dataframe (pd.DataFrame): The data source.
        x_var (str): The column name for the x-axis.
        y_var (str): The column name for the y-axis.
        color_var (str, optional): The column name for coloring the points.
        Defaults to None.

    Returns:
        None. Displays a scatter plot.
    """
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=dataframe,
        x=x_var,
        y=y_var,
        hue=color_var,
        palette="viridis",
        alpha=0.7)
    plt.title(
        f"Relational Plot: {x_var.capitalize()} vs {y_var.capitalize()}",
        fontsize=14)
    plt.xlabel(x_var.capitalize(), fontsize=12)
    plt.ylabel(y_var.capitalize(), fontsize=12)
    plt.grid(True)
    if color_var is not None:
        plt.legend(title=color_var)
    plt.tight_layout()
    plt.show()


def categorical_top_genres_plot(dataframe, top_n=10):
    """
    Create a bar plot for the top N genres in the dataset.

    Parameters:
        dataframe (pd.DataFrame): The data source.
        top_n (int, optional): Number of top genres to display. Defaults to 10.

    Returns:
        None. Displays a bar plot.
    """
    plt.figure(figsize=(12, 8))
    top_genres = dataframe["genre"].value_counts().nlargest(top_n)
    sns.barplot(x=top_genres.values, y=top_genres.index, palette="Spectral")
    plt.title(f"Top {top_n} Spotify Genres", fontsize=16)
    plt.xlabel("Number of Songs", fontsize=14)
    plt.ylabel("Genre", fontsize=14)
    # Annotate bar values
    for index, value in enumerate(top_genres.values):
        plt.text(
            value,
            index,
            f" {value}",
            va="center",
            fontsize=12,
            color="black")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_heatmap(dataframe):
    """
    Generate a heatmap of the correlation matrix for numeric features in the
    dataframe.

    Parameters:
        dataframe (pd.DataFrame): The data source.

    Returns:
        None. Displays a heatmap.
    """
    plt.figure(figsize=(12, 8))
    # Calculate correlation for numeric features only
    numeric_df = dataframe.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_distribution(dataframe):
    """
    Plot distribution for all columns in the dataframe.

    For numeric columns, display histograms with KDE; for categorical columns,
    display count plots.

    Parameters:
        dataframe (pd.DataFrame): The data source.

    Returns:
        None. Displays multiple subplots of distributions.
    """
    cols = dataframe.columns
    num_cols = len(cols)
    n_rows = (num_cols + 1) // 2
    plt.figure(figsize=(15, n_rows * 5))
    for idx, col in enumerate(cols, 1):
        plt.subplot(n_rows, 2, idx)
        if dataframe[col].dtype in ["float64", "int64"]:
            sns.histplot(dataframe[col].dropna(), kde=True, bins=30)
            plt.title(f"Distribution of Numeric Variable: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
        else:
            sns.countplot(
                y=dataframe[col],
                order=dataframe[col].value_counts().index[:15]
            )
            plt.title(f"Distribution of Categorical Variable: {col}")
            plt.xlabel("Count")
            plt.ylabel(col)
    plt.tight_layout()
    plt.show()


def plot_pairplot(dataframe, features):
    """
    Generate a pairplot for selected numeric features.

    Parameters:
        dataframe (pd.DataFrame): The data source.
        features (list): A list of column names to include in the pairplot.

    Returns:
        None. Displays a pairplot.
    """
    sns.pairplot(dataframe[features], diag_kind="kde", kind="scatter")
    plt.suptitle("Pairplot of Selected Numeric Variables", y=1.02)
    plt.tight_layout()
    plt.show()


# Clustering Functions
def elbow_method(dataframe, features, max_k=10):
    """
    Use the Elbow Method to determine the optimal number of clusters for
    KMeans clustering.

    Parameters:
        dataframe (pd.DataFrame): The data source.
        features (list): A list of column names to use for clustering.
        max_k (int, optional): The maximum number of clusters to try.
        Defaults to 10.

    Returns:
        None. Displays a plot of the sum of squared distances (SSE) for
        different values of k.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe[features])
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), sse, "bo-", markersize=8)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Sum of squared distances (SSE)")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.show()


def silhouette_method(dataframe, features, max_k=10):
    """
    Use the Silhouette Method to evaluate the quality of clustering for
    different numbers of clusters.

    Parameters:
        dataframe (pd.DataFrame): The data source.
        features (list): A list of column names to use for clustering.
        max_k (int, optional): The maximum number of clusters to try. Defaults
        to 10.

    Returns:
        None. Displays a plot of silhouette scores for different values of k.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe[features])
    silhouette_avg = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg.append(silhouette_score(X_scaled, cluster_labels))
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), silhouette_avg, "ro-", markersize=8)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")
    plt.tight_layout()
    plt.show()


def k_means_clustering(dataframe, features, n_clusters=4):
    """
    Apply K-Means clustering to the dataframe using selected features and add
    the resulting cluster labels to the dataframe.

    Parameters:
        dataframe (pd.DataFrame): The data source.
        features (list): A list of column names to use for clustering.
        n_clusters (int, optional): The number of clusters to form. Defaults
        to 4.

    Returns:
        pd.DataFrame: The dataframe with an additional 'cluster' column.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dataframe["cluster"] = kmeans.fit_predict(X_scaled)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=features[0],
        y=features[1],
        hue="cluster",
        data=dataframe,
        palette="Set1")
    plt.title("K-Means Clustering")
    plt.tight_layout()
    plt.show()
    return dataframe


# Polynomial Fitting Functions
def polynomial_fit(dataframe, x_feature, y_feature, degree=3):
    """
    Fit a polynomial of a given degree to two variables and plot the fit.

    Parameters:
        dataframe (pd.DataFrame): The data source.
        x_feature (str): The column name for the independent variable.
        y_feature (str): The column name for the dependent variable.
        degree (int, optional): The degree of the polynomial. Defaults to 3.

    Returns:
        np.poly1d: The polynomial model.
    """
    x = dataframe[x_feature]
    y = dataframe[y_feature]
    # Polynomial fitting using numpy Polynomial
    coefs = Polynomial.fit(x, y, degree).convert().coef
    p = np.poly1d(coefs[::-1])
    # Generate predictions
    x_pred = np.linspace(x.min(), x.max(), 200)
    y_pred = p(x_pred)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Data points", alpha=0.6)
    plt.plot(x_pred, y_pred, color="red", label=f"{
             degree}-degree Polynomial Fit")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"Polynomial Fit ({degree}-degree)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return p


def predict_with_confidence(
    poly_model, dataframe, x_feature, y_feature, confidence=0.95
):
    """
    Generate predictions with a polynomial model, calculate confidence
    intervals, and plot the results.

    Parameters:
        poly_model (np.poly1d): The polynomial model.
        dataframe (pd.DataFrame): The data source.
        x_feature (str): The column name for the independent variable.
        y_feature (str): The column name for the dependent variable.
        confidence (float, optional): The confidence level for the interval.
        Defaults to 0.95.

    Returns:
        None. Displays a plot with the polynomial fit and confidence intervals.
    """
    x = dataframe[x_feature]
    y = dataframe[y_feature]
    x_pred = np.linspace(x.min(), x.max(), 200)
    y_pred = poly_model(x_pred)
    n = len(x)
    m = np.mean(x)
    # Calculate standard error using MSE
    se = np.sqrt(mean_squared_error(y, poly_model(x)))
    t_score = t.ppf((1 + confidence) / 2, n - 2)
    ci = t_score * se * np.sqrt(1 / n + (x_pred - m)
                                ** 2 / np.sum((x - m) ** 2))
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Data points", alpha=0.6)
    plt.plot(x_pred, y_pred, color="red", label="Polynomial Fit")
    plt.fill_between(
        x_pred,
        y_pred - ci,
        y_pred + ci,
        color="grey",
        alpha=0.4,
        label=f"{int(confidence * 100)}% Confidence Interval",
    )
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title("Polynomial Fit with Confidence Interval")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Statistical Moments Function
def statistical_moments(dataframe, features):
    """
    Calculate and display statistical moments (mean, variance, skewness,
                                               kurtosis) for a set of features.

    Parameters:
        dataframe (pd.DataFrame): The data source.
        features (list): A list of column names for which to calculate moments.

    Returns:
        None. Prints a DataFrame with statistical moments.
    """
    moments_list = []
    for feature in features:
        moments_list.append(
            {
                "Feature": feature,
                "Mean": np.mean(dataframe[feature]),
                "Variance": np.var(dataframe[feature]),
                "Skewness": skew(dataframe[feature]),
                "Kurtosis": kurtosis(dataframe[feature]),
            }
        )
    moments_df = pd.DataFrame(moments_list)
    print("Statistical Moments:")
    print(moments_df, "\n")


# Main execution function
def main():
    """
    Main function to execute the data analysis workflow.

    Steps:
      1. Load and clean data from 'data.csv'.
      2. Generate various visualizations including relational plots, bar
      plots, heatmaps, distributions, and pairplots.
      3. Determine optimal clusters using Elbow and Silhouette methods.
      4. Apply K-Means clustering and visualize the results.
      5. Fit a polynomial model to the data and plot predictions with
      confidence intervals.
      6. Calculate and display statistical moments for selected features.

    Returns:
        None.
    """
    # Load and clean the data from the provided CSV file
    df = load_and_clean_data("data.csv")

    # Visualization: Relational Plot (e.g., danceability vs energy colored by
    # valence)
    relational_plot(df, "danceability", "energy", "valence")

    # Visualization: Top Genres (ensure 'genre' column exists in your data)
    if "genre" in df.columns:
        categorical_top_genres_plot(df, top_n=10)
    else:
        print("Column 'genre' not found in data. Skipping top genres plot.")

    # Visualization: Correlation Heatmap
    plot_heatmap(df)

    # Visualization: Distribution Plots for all variables
    plot_distribution(df)

    # Visualization: Pairplot of Selected Numeric Variables
    important_features = [
        "danceability",
        "energy",
        "valence",
        "tempo",
        "loudness"]
    # Check if these features exist before plotting
    available_features = [
        feat for feat in important_features if feat in df.columns]
    if len(available_features) >= 2:
        plot_pairplot(df, available_features)
    else:
        print("Not enough numeric features available for pairplot.")

    # Clustering: Determine optimal clusters using Elbow and Silhouette methods
    features_for_clustering = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    # Ensure features exist in the dataset
    features_for_clustering = [
        feat for feat in features_for_clustering if feat in df.columns
    ]
    if len(features_for_clustering) >= 2:
        elbow_method(df, features_for_clustering, max_k=10)
        silhouette_method(df, features_for_clustering, max_k=10)
        # Apply K-Means clustering (using 4 clusters as default)
        df = k_means_clustering(df, features_for_clustering, n_clusters=4)
    else:
        print("Not enough features available for clustering.")

    # Polynomial Fitting: Fit and plot a polynomial for danceability vs energy
    if all(col in df.columns for col in ["danceability", "energy"]):
        poly_model = polynomial_fit(df, "danceability", "energy", degree=3)
        predict_with_confidence(
            poly_model, df, "danceability", "energy", confidence=0.95
        )
    else:
        print("Required columns for polynomial fitting not found.")

    # Statistical Moments: Compute and display moments for clustering features
    if features_for_clustering:
        statistical_moments(df, features_for_clustering)
    else:
        print("No features available for statistical moment calculation.")


# Entry Point
if __name__ == "__main__":
    main()
