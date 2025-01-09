
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def feature_selection(X, y, k=10):
    selector = SelectKBest(f_classif, k=k)
    selected_features = selector.fit_transform(X, y)
    return selected_features

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca
