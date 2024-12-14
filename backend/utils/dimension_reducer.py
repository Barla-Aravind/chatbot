import numpy as np
from sklearn.decomposition import PCA

class DimensionReducer:
    @staticmethod
    def reduce_dimensions(embeddings, target_dimensions=128):
        """
        Reduce high-dimensional embeddings using PCA
        
        Args:
            embeddings (np.ndarray): Input embeddings
            target_dimensions (int): Desired output dimension
        
        Returns:
            np.ndarray: Reduced dimension embeddings
        """
        try:
            # Ensure target dimensions is not larger than original
            target_dimensions = min(target_dimensions, embeddings.shape[1])
            
            # Initialize and fit PCA
            pca = PCA(n_components=target_dimensions)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            return reduced_embeddings
        
        except Exception as e:
            print(f"Dimension Reduction Error: {e}")
            raise
    
    @staticmethod
    def variance_explained_ratio(embeddings, max_components=10):
        """
        Calculate variance explained by different number of components
        
        Args:
            embeddings (np.ndarray): Input embeddings
            max_components (int): Maximum components to analyze
        
        Returns:
            np.ndarray: Cumulative variance explained
        """
        try:
            pca = PCA()
            pca.fit(embeddings)
            
            # Calculate cumulative variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            return cumulative_variance[:max_components]
        
        except Exception as e:
            print(f"Variance Analysis Error: {e}")
            raise