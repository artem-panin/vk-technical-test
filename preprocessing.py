import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.model_selection import train_test_split
import attr


@attr.s()
class Preprocessor:
    validate: bool = attr.ib(default=False)
    target: str = attr.ib(default='target')
    idf_normalize: bool = attr.ib(default=False)
    test_size: float = attr.ib(default=0.2)
    random_state: int = attr.ib(default=0)
    csv_path: str = attr.ib(default='data/friends_dataset')

    def __attrs_post_init__(self):
        """
        Data reading and preprocessing, creating sparse matrix of interaction
        """
        self.df = self._read_csv()
        if self.validate:
            self.X_train, self.X_test = train_test_split(self.df, test_size=self.test_size, shuffle=False,
                                                         random_state=self.random_state)
            self.train_matrix = self._create_interaction_matrix(self.X_train, target_col=self.target)
        else:
            self.train_matrix = self._create_interaction_matrix(self.df, target_col=self.target)
        self.train_csr = csr_matrix(self.train_matrix.values)

    def _read_csv(self) -> pd.DataFrame:
        """
        Read and preprocess friends dataframe
        Returns
        -------
        df : pd.DataFrame
            Dataframe with right columns and custom target
        """
        df = pd.read_csv('data/friends_dataset', header=None)
        df.columns = ['uid1', 'uid2', 'time', 'intensity']
        df = df.groupby(['uid1', 'uid2'])[['time', 'intensity']].mean().reset_index()
        df['time'] = (df['time'] - df['time'].min()) / (df['time'].max() - df['time'].min())
        df['intensity'] = (df['intensity'] - df['intensity'].min()) / (df['intensity'].max() - df['intensity'].min())
        df['target'] = 1 + df['intensity']
        df = df.sort_values(by='time', ascending=False).reset_index(drop=True)
        return df

    def _create_interaction_matrix(self, df, target_col='target') -> pd.DataFrame:
        """
        Created square iteration matrix (NaN values updated using transposed values)
        Parameters
        ----------
        df : pd.DataFrame
            Users interaction history dataset
        target_col : str
            Label of target column (interaction intensity, time)
        Returns
        -------
        interaction_matrix : pd.Dataframe
            Matrix of full users interactions
        """
        interaction_matrix = df[['uid1', 'uid2', target_col]].pivot(index='uid1', columns='uid2')
        interaction_matrix.columns = interaction_matrix.columns.droplevel(0)
        full_index = interaction_matrix.index.union(interaction_matrix.columns).sort_values()
        interaction_matrix = interaction_matrix.reindex(index=full_index, columns=full_index)
        interaction_matrix.update(interaction_matrix.T)
        if self.idf_normalize:
            vc = df[['uid1', 'uid2']].stack().value_counts(normalize=True)
            idf = np.log(1 / vc).sort_index()
            interaction_matrix = interaction_matrix.multiply(idf, axis=1)
        return interaction_matrix.fillna(0)

    def cache_data(self, name=''):
        """
        Cache data for next model usage
        Parameters
        ----------
        name : str
            Substring for saving idf-normalized matrix using new name
        """
        if self.idf_normalize:
            name = '_idf'
        if self.validate:
            save_npz(f'data/validate/train_csr{name}.npz', self.train_csr)
            self.train_matrix.index.to_series().to_csv(f'data/validate/train_index_map.csv')
            self.X_train.to_csv(f'data/validate/train_df.csv', index=False)
            self.X_test.to_csv(f'data/validate/test_df.csv', index=False)
        else:
            save_npz(f'data/prod/csr{name}.npz', self.train_csr)
            self.train_matrix.index.to_series().to_csv(f'data/prod/index_map.csv')
            self.df.to_csv(f'data/prod/df.csv', index=False)
