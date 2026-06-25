import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

class HRPEngine:
    def __init__(self, returns_df):
        self.returns = returns_df
        self.tickers = returns_df.columns.tolist()
        self.cov = returns_df.cov()
        self.corr = returns_df.corr()
        self.weights = None
        self.linkage = None

    def compute_distance_matrix(self):
        """Transforms correlation matrix into mathematical distance matrix."""
        # Handle potential rounding errors ensuring correlations are within [-1, 1]
        corr_clipped = np.clip(self.corr.values, -1.0, 1.0)
        dist = np.sqrt(0.5 * (1.0 - corr_clipped))
        return dist

    def compute_quasi_diag(self, link):
        """Sorts tickers to cluster similar assets together (Quasi-Diagonalization)."""
        return sch.to_tree(link, rd=False).pre_order()

    def get_cluster_var(self, cov, cluster_items):
        """Calculates variance of an asset cluster using inverse-variance allocation."""
        cov_slice = cov.iloc[cluster_items, cluster_items]
        inv_var = 1.0 / np.diag(cov_slice.values)
        weights = inv_var / np.sum(inv_var)
        cluster_var = np.dot(np.dot(weights, cov_slice.values), weights)
        return cluster_var

    def get_rec_bisection(self, cov, sort_items):
        """Recursively splits clusters and assigns weights based on aggregate variance."""
        weights = pd.Series(1.0, index=self.tickers)
        cluster_list = [sort_items]

        while len(cluster_list) > 0:
            # Internal loop to segment clusters
            cluster_list = [c[i:j] for c in cluster_list for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
            
            for i in range(0, len(cluster_list), 2):
                cluster_1 = cluster_list[i]
                cluster_2 = cluster_list[i + 1]
                
                # Get map positions relative to original dataframe columns
                c_items_1 = [self.tickers.index(t) for t in cluster_1]
                c_items_2 = [self.tickers.index(t) for t in cluster_2]
                
                var_1 = self.get_cluster_var(cov, c_items_1)
                var_2 = self.get_cluster_var(cov, c_items_2)
                
                alpha_1 = 1.0 - var_1 / (var_1 + var_2)
                alpha_2 = 1.0 - alpha_1
                
                weights[cluster_1] *= alpha_1
                weights[cluster_2] *= alpha_2
        return weights

    def allocate(self):
        """Executes full HRP allocation pipelining."""
        dist = self.compute_distance_matrix()
        
        # Scipy condensed distance matrix conversion
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(dist, checks=False)
        
        # Hierarchical Single Linkage Clustering
        self.linkage = sch.linkage(condensed_dist, method='single')
        
        # Sort tickers via quasi-diagonalization
        order = self.compute_quasi_diag(self.linkage)
        sorted_tickers = [self.tickers[i] for i in order]
        
        # Allocate weights via recursive bisection
        self.weights = self.get_rec_bisection(self.cov, sorted_tickers)
        return self.weights
