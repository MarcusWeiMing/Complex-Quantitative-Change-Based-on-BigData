from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import numpy as np

def feature_selection(X, y, k=10):
    """
    Selects the top k features from the dataset X based on the statistical test between
    the features and the target variable y using SelectKBest.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input feature matrix, where each row represents a sample, and each column represents a feature.
    
    y : array-like, shape (n_samples,)
        The target vector, with class labels or regression values, depending on the problem.
    
    k : int, optional (default=10)
        The number of top features to select based on their statistical score.
        If k exceeds the number of features in X, all features will be selected.
    
    Returns:
    selected_features : array-like, shape (n_samples, k)
        The selected top k features from the input dataset X.
    
    Example:
    >>> X_selected = feature_selection(X_train, y_train, k=5)
    """
    # Check if k is less than the number of features in X
    if k > X.shape[1]:
        raise ValueError(f"k cannot be greater than the number of features ({X.shape[1]}) in the dataset.")
    
    # Initialize the SelectKBest selector with the F-value score function for classification tasks
    selector = SelectKBest(f_classif, k=k)
    
    # Fit the selector and transform the feature matrix X to select top k features
    selected_features = selector.fit_transform(X, y)
    
    return selected_features


def apply_pca(X, n_components=2, scale_data=False):
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the dataset X.
    PCA transforms the dataset into a set of orthogonal components that maximize the variance.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input feature matrix to be transformed.
    
    n_components : int, optional (default=2)
        The number of principal components to retain. If set to None, all components are retained.
    
    scale_data : bool, optional (default=False)
        Whether to standardize (scale) the features before applying PCA.
        It is recommended to scale data if the features have different units.

    Returns:
    X_pca : array-like, shape (n_samples, n_components)
        The transformed dataset with reduced dimensions.
    
    explained_variance_ratio : array, shape (n_components,)
        The percentage of variance explained by each of the selected components.

    Example:
    >>> X_pca, variance_ratio = apply_pca(X_train, n_components=3, scale_data=True)
    """
    # Optional: Scale the data if needed
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Initialize PCA with the desired number of components
    pca = PCA(n_components=n_components)
    
    # Fit the PCA model to the data and transform X to lower dimensions
    X_pca = pca.fit_transform(X)
    
    # Return the transformed data and explained variance ratio
    return X_pca, pca.explained_variance_ratio_


# Additional example usage for both functions

# Example dataset (use real data here for practical purposes)
X_example = np.random.rand(100, 10)  # 100 samples, 10 features
y_example = np.random.randint(0, 2, size=100)  # Binary target variable (for classification)

# Feature Selection: Selecting top 5 features based on the f_classif score function
X_selected = feature_selection(X_example, y_example, k=5)

# PCA: Reducing dimensionality to 2 components and scaling the features
X_pca, variance_ratio = apply_pca(X_example, n_components=2, scale_data=True)

print("Selected Features Shape:", X_selected.shape)
print("PCA Transformed Data Shape:", X_pca.shape)
print("Explained Variance Ratio:", variance_ratio)
