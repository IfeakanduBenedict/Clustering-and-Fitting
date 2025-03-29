# Essential libraries for data handling, visualization, and modeling
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from scipy.stats import skew, kurtosis

# Load and Inspect Dataset
spotify_df = pd.read_csv('Spotify_Song_Attributes.csv')

spotify_df.head()

spotify_df.tail()

spotify_df.describe()

spotify_df.info()

spotify_df.isnull().sum()

spotify_df.duplicated().sum()

spotify_df.columns

# Dropping irrelevant columns
columns_to_drop = ['uri', 'track_href', 'analysis_url', 'id', 'type']

# Check if columns exist before dropping
existing_columns = spotify_df.columns
columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

spotify_df.drop(columns=columns_to_drop, inplace=True)

# Checking to know if irrelevant columns dropped
spotify_df.columns

spotify_df.head()

# Removing rows with missing values
spotify_df.dropna(inplace=True)

# Define relational plot function with color option
def relational_plot(dataframe, x_var, y_var, color_var=None):
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=dataframe, x=x_var, y=y_var, hue=color_var, palette='viridis', alpha=0.7)
    plt.title(f'Relational Plot: {x_var.capitalize()} vs {y_var.capitalize()}', fontsize=14)
    plt.xlabel(x_var.capitalize(), fontsize=12)
    plt.ylabel(y_var.capitalize(), fontsize=12)
    plt.grid(True)
    plt.legend(title=color_var)
    plt.tight_layout()
    plt.show()

relational_plot(spotify_df, 'danceability', 'energy', 'valence')

# Define categorical plot function with visual enhancements
def categorical_top_genres_plot(dataframe, top_n=10):
    plt.figure(figsize=(12, 8))
    top_genres = dataframe['genre'].value_counts().nlargest(top_n)
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='Spectral')
    plt.title(f'Top {top_n} Spotify Genres', fontsize=16)
    plt.xlabel('Number of Songs', fontsize=14)
    plt.ylabel('Genre', fontsize=14)

    # Adding annotations to enhance visual insight
    for index, value in enumerate(top_genres.values):
        plt.text(value, index, f' {value}', va='center', fontsize=12, color='black')

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

categorical_top_genres_plot(spotify_df, top_n=10)

# Define correlation heatmap function
def plot_heatmap(dataframe):
    plt.figure(figsize=(12,8))
    # Calculate correlation for numeric features only
    numeric_df = dataframe.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

plot_heatmap(spotify_df)

# Defining a plotting function for the variables.
def plot_distribution(dataframe):
    cols = dataframe.columns
    num_cols = len(cols)
    n_rows = (num_cols + 1) // 2

    plt.figure(figsize=(15, n_rows * 5))
    for idx, col in enumerate(cols, 1):
        plt.subplot(n_rows, 2, idx)
        if dataframe[col].dtype in ['float64', 'int64']:
            sns.histplot(dataframe[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of Numeric Variable: {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        else:
            sns.countplot(y=dataframe[col], order=dataframe[col].value_counts().index[:15])
            plt.title(f'Distribution of Categorical Variable: {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# Plot distribution for all variables using the defined function
plot_distribution(spotify_df)

important_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness']

# Define pairplot function with selected important features
def plot_pairplot(dataframe, important_features):
    sns.pairplot(dataframe[important_features], diag_kind='kde', kind='scatter')
    plt.suptitle("Pairplot of Selected Numeric Variables", y=1.02)
    plt.show()

plot_pairplot(spotify_df, important_features)

# Elbow Method for optimal K
def elbow_method(dataframe, features, max_k=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe[features])

    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_k + 1), sse, 'bo-', markersize=8)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances (SSE)')
    plt.title('Elbow Method')
    plt.show()

# Define the features for clustering before calling elbow_method
features_for_clustering = ['danceability', 'energy', 'loudness', 'speechiness',
                           'acousticness', 'instrumentalness', 'liveness',
                           'valence', 'tempo'] # Add or remove features as needed

elbow_method(spotify_df, features_for_clustering, max_k=10)

# Silhouette Method for optimal K
def silhouette_method(dataframe, features, max_k=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe[features])

    silhouette_avg = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg.append(silhouette_score(X_scaled, cluster_labels))

    plt.figure(figsize=(8,5))
    plt.plot(range(2, max_k + 1), silhouette_avg, 'ro-', markersize=8)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.show()

silhouette_method(spotify_df, features_for_clustering, max_k=10)

# K-Means Clustering on audio features
def k_means_clustering(dataframe, features, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dataframe['cluster'] = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features[0], y=features[1], hue='cluster', data=dataframe, palette='Set1')
    plt.title('K-Means Clustering')
    plt.show()
    return dataframe

spotify_df = k_means_clustering(spotify_df, features_for_clustering, n_clusters=4)

# Polynomial fitting function
def polynomial_fit(dataframe, x_feature, y_feature, degree=3):
    x = dataframe[x_feature]
    y = dataframe[y_feature]

    # Import Polynomial from numpy.polynomial.polynomial
    from numpy.polynomial.polynomial import Polynomial # Importing the Polynomial class

    # Polynomial fitting
    coefs = Polynomial.fit(x, y, degree).convert().coef
    p = np.poly1d(coefs[::-1])

    # Generate predictions
    x_pred = np.linspace(x.min(), x.max(), 200)
    y_pred = p(x_pred)

    plt.figure(figsize=(10,6))
    plt.scatter(x, y, label='Data points', alpha=0.6)
    plt.plot(x_pred, y_pred, color='red', label=f'{degree}-degree Polynomial fit')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f'Polynomial Fit ({degree}-degree)')
    plt.legend()
    plt.show()

    return p

# Polynomial fitting example
poly_model = polynomial_fit(spotify_df, 'danceability', 'energy', degree=3)

# Making predictions with confidence intervals and plotting
def predict_with_confidence(poly_model, dataframe, x_feature, y_feature, confidence=0.95):
    x = dataframe[x_feature]
    y = dataframe[y_feature]
    x_pred = np.linspace(x.min(), x.max(), 200)
    y_pred = poly_model(x_pred)

    n = len(x)
    m = np.mean(x)
    se = np.sqrt(mean_squared_error(y, poly_model(x)))
    t_score = t.ppf((1 + confidence) / 2, n - 2)
    ci = t_score * se * np.sqrt(1/n + (x_pred - m)**2 / np.sum((x - m)**2))

    plt.figure(figsize=(10,6))
    plt.scatter(x, y, label='Data points', alpha=0.6)
    plt.plot(x_pred, y_pred, color='red', label='Polynomial Fit')
    plt.fill_between(x_pred, y_pred - ci, y_pred + ci, color='grey', alpha=0.4, label=f'{int(confidence*100)}% Confidence Interval')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title('Polynomial Fit with Confidence Interval')
    plt.legend()
    plt.show()

# Plot prediction with confidence interval
predict_with_confidence(poly_model, spotify_df, 'danceability', 'energy', confidence=0.95)

def statistical_moments(dataframe, features):
    moments_list = []  # Collect moments as a list of dictionaries

    for feature in features:
        mean_val = np.mean(dataframe[feature])
        var_val = np.var(dataframe[feature])
        skew_val = skew(dataframe[feature])
        kurtosis_val = kurtosis(dataframe[feature])

        # Add feature moments to list
        moments_list.append({
            'Feature': feature,
            'Mean': mean_val,
            'Variance': var_val,
            'Skewness': skew_val,
            'Kurtosis': kurtosis_val
        })

    # Create DataFrame from the list of dictionaries
    moments_df = pd.DataFrame(moments_list)
    print(moments_df)

# Compute and display statistical moments
statistical_moments(spotify_df, features_for_clustering)
