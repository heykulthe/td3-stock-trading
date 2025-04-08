import numpy as np
import pandas as pd
import logging

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('td3-stock-trading')

class DimensionReducer:
    def __init__(self, method='pca', n_components=20, feature_columns=None):

        self.method = method
        self.n_components = n_components
        self.feature_columns = feature_columns
        self.pca = None
        self.scaler = None

    def fit_transform(self, data):

        if self.method == 'pca':
            return self._apply_pca(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _apply_pca(self, data):

        time_col = None
        if 'time' in data.columns:
            time_col = data['time']
            data = data.drop('time', axis=1)


        self.pca = PCA(n_components=self.n_components)
        reduced_data = self.pca.fit_transform(data)

        reduced_df = pd.DataFrame(
            reduced_data,
            columns=[f'pc{i + 1}' for i in range(self.n_components)]
        )

        if time_col is not None:
            reduced_df['time'] = time_col.values

        logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        return reduced_df


    def transform(self, data):

        if self.method == 'pca':
            # Keep time column separate if it exists
            time_col = None
            if 'time' in data.columns:
                time_col = data['time']
                data = data.drop('time', axis=1)

            # Scale and transform
            scaled_data = self.scaler.transform(data)
            reduced_data = self.pca.transform(scaled_data)

            # Convert to DataFrame
            reduced_df = pd.DataFrame(
                reduced_data,
                columns=[f'pc{i + 1}' for i in range(self.n_components)]
            )

            # Add time column back if it existed
            if time_col is not None:
                reduced_df['time'] = time_col.values

            return reduced_df
        elif self.method == 'select':
            return data[self.feature_columns]
        else:
            raise ValueError(f"Unknown method: {self.method}")