import numpy as np
from sklearn.neighbors import NearestNeighbors
from skdim._commonfuncs import GlobalEstimator

class WODBalls(GlobalEstimator):
    
    def __init__(self, k):
        self.k = k
    
    def fit_transform(self, X, y=None):
        knn_matrix, indices = self.k_nearest_neighbors_matrix(X, self.k)
        all_indices_per_row, ratios = self.calculate_ratios(indices, self.k)
        average_ratio = sum(ratios) / len(ratios)
        return -np.log2(average_ratio)
    
    def k_nearest_neighbors_matrix(self, data, k):
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
        
        distances, indices = nbrs.kneighbors(data)
        
        n_points = data.shape[0]
        knn_matrix = np.zeros((n_points, n_points), dtype=int)
        
        for i in range(n_points):
            knn_matrix[i, indices[i]] = 1
        "print(knn_matrix)"
        return knn_matrix, indices
    
    def calculate_ratios(self, indices, k):
        all_indices_per_row = []
        ratios = []
        for i in range(len(indices)):
            current_knn = indices[i][0:]
            row_indices = set(current_knn)
            
            for idx in current_knn:
                knn_indices = indices[idx][0:]
                row_indices.update(knn_indices)
            
            unique_indices = sorted(row_indices)
            all_indices_per_row.append(unique_indices)
            
            ratio = (k + 1) / len(unique_indices)
            ratios.append(ratio)
            "print(unique_indices)"
        
        return all_indices_per_row, ratios
    
    