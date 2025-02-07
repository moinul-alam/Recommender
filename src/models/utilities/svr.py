import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

def plot_explained_variance(user_item_matrix, max_components=1000):
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    svd.fit(user_item_matrix)
    explained_variance = svd.explained_variance_ratio_.cumsum()

    plt.plot(explained_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Components')
    plt.show()

    # Example: Choose the number of components that explain 95% of the variance
    n_components = np.argmax(explained_variance >= 0.70) + 1
    return n_components

# Example usage
file_path = 'E:/recommender-pipeline/data/collaborative/v2/2. processed/user_item_matrix.pkl'
# Open the pickle file and load the object
with open(file_path, 'rb') as file:
    user_item_matrix = pickle.load(file)
    
n_components = plot_explained_variance(user_item_matrix)
print(f"Optimal number of components: {n_components}")