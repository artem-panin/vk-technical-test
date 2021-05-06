# external
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from cached_property import cached_property
import networkx as nx
import attr

# built-in
from collections import Counter
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")


@attr.s()
class BaseModel:
    random_state: int = attr.ib(default=0)
    validate: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        self._load_data()
        self._set_uids2indiced_map()

    @staticmethod
    def build_graph(path) -> nx.Graph:
        """
        Create graph representation of dataset
        Parameters
        ----------
        path : str
            Relative path to csv file
        """
        df = pd.read_csv(path, usecols=['uid1', 'uid2', 'target'])
        df.columns = ['uid1', 'uid2', 'weight']
        df['weight'] = 1 / df['weight']
        g = nx.from_pandas_edgelist(df, source='uid1', target='uid2', edge_attr=['weight'])
        return g

    def _load_data(self, name=''):
        """
        Load data from cache (cached by Preprocessor module)
        Parameters
        ----------
        name : str
            Substring for loading idf-normalized matrix using new name
        """
        if self.validate:
            self.train_csr = load_npz(f'data/validate/train_csr{name}.npz')
            self.index_map = pd.read_csv(f'data/validate/train_index_map.csv', index_col=0, header=None)[1].values
            self.train_df = pd.read_csv(f'data/validate/train_df.csv')
            self.train_graph = self.build_graph(f'data/validate/train_df.csv')
        else:
            self.train_csr = load_npz(f'data/prod/csr{name}.npz')
            self.index_map = pd.read_csv(f'data/prod/index_map.csv', index_col=0, header=None)[1].values
            self.train_df = pd.read_csv(f'data/prod/df.csv')
            self.train_graph = self.build_graph(f'data/prod/df.csv')
        self.neighbors = self.get_neighbors_dict(self.train_graph)
        self.second_level = self.get_second_level_neighbors(self.train_graph)


    def _get_uids(self) -> list:
        """
        Returns
        -------
        uid_list : list
            Sorted list of all users in dataset
        """
        return sorted(self.train_graph.nodes)

    def _set_uids2indiced_map(self):
        """
        Set mapping between real indices and np.arrange()
        """
        uids = self._get_uids()
        self.ind2uid = pd.Series(uids, index=range(len(uids)))
        self.uid2ind = pd.Series(range(len(uids)), index=uids)

    def _get_common_friends(self, uid1, uid2) -> set:
        """
        Parameters
        ----------
        uid1 : int
            First user id
        uid2 : int
            Second user id
        Returns
        -------
        common_friends : set
            Set of user's common friends
        """
        uid1_neighbors = set(self.train_graph.neighbors(uid1))
        uid2_neighbors = set(self.train_graph.neighbors(uid2))
        return uid1_neighbors & uid2_neighbors

    def _get_kth_level_neighbors(self, graph, k) -> dict:
        k_level_neighbors = {node: [neighbor for neighbor, level in
                                    nx.single_source_shortest_path_length(graph, node, cutoff=k).items()
                                    if level == k] for node in sorted(graph.nodes)}
        return k_level_neighbors

    def get_second_level_neighbors(self, graph) -> dict:
        neighbors_dict = {}
        for node in tqdm(sorted(graph.nodes)):
            neighbors = set(graph.neighbors(node))
            second_level = []
            for i in neighbors:
                second_level += list(graph.neighbors(i))
            second_level = Counter(second_level)
            del second_level[node]
            for word in list(second_level):
                if word in neighbors:
                    del second_level[word]
            neighbors_dict[node] = second_level
        return neighbors_dict

    def get_neighbors_dict(self, graph) -> dict:
        """
        Computes friends first-level connections in dataset
        Parameters
        ----------
        df : pd.DataFrame
            Users interaction history dataset
        Returns
        -------
        friends_dict_train : dictionary
            Dictionary with users as keys and list of his friends as values
        """
        neighbors = {node: list(graph.neighbors(node)) for node in sorted(graph.nodes)}
        return neighbors

    @cached_property
    def _find_popular(self) -> list:
        """
        Computes most popular users in dataset
        Returns
        -------
        popular : np.array
            Ordered array of popular users
        """
        popular = self.train_df[['uid1', 'uid2']].stack().value_counts()
        return list(popular.index)

    def _find_nearest_neighbors(self, user, k, method='BFS') -> list:
        """
        Computes k nearest potential friends for users in list
        Parameters
        ----------
        user : np.array
            User's uids who needs recommendations
        k : int
            The maximum number of predicted elements
        method : int, optional
            Calculation method: Dijkstra and BFS for graphs and custom approach
        Returns
        -------
        recommendations : np.array
            Ordered list with recommended users uids
        """
        if user in self.train_graph.nodes:
            nearest_neighbors = []
            if method == 'BFS':
                for neighbor, level in iter((nx.single_source_shortest_path_length(
                        self.train_graph, user, cutoff=2)).items()):
                    if level > 1:
                        nearest_neighbors.append(neighbor)
                    if len(nearest_neighbors) == k:
                        return nearest_neighbors
            elif method == 'dijkstra':
                for neighbor in iter((nx.single_source_dijkstra(self.train_graph, user, cutoff=3))):
                    if neighbor not in self.neighbors[user]:
                        nearest_neighbors.append(neighbor)
                    if len(nearest_neighbors) == k:
                        return nearest_neighbors
            elif method == 'custom':
                for friend in self.neighbors[user]:
                    for his_friend in self.neighbors[friend]:
                        if his_friend not in self.neighbors[user] and his_friend != user:
                            nearest_neighbors.append(his_friend)
                nearest_neighbors = np.array(Counter(nearest_neighbors).most_common(k))
                if len(nearest_neighbors) > 0:
                    nearest_neighbors = nearest_neighbors[:, 0]
                nearest_neighbors = list(nearest_neighbors[:k])
                if len(nearest_neighbors) == k:
                    return nearest_neighbors
            elif method == 'super_custom':
                pred = np.array(self.second_level[user].most_common()[:5])
                if len(pred) > 0:
                    pred = pred[:, 0]
                if len(pred) == k:
                    return pred
            popular = self._find_popular[:k]
            for i in range(k - len(nearest_neighbors)):
                nearest_neighbors.append(popular[i])
            return nearest_neighbors
        else:
            return self._find_popular[:k]

    def fit(self):
        raise NotImplementedError

    def predict(self, users_list, k=5):
        raise NotImplementedError

    def to_cache(self, path):
        """
        Serialize model
        Parameters
        ----------
        path : str
            Relative path to serialized file
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=4)


@attr.s()
class HeuristicsModel(BaseModel):
    method: str = attr.ib(default='custom')
    logging.basicConfig(filename="HeuristicsModel.log", level=logging.INFO)
    logger = logging.getLogger('HeuristicsModel')

    def fit(self):
        pass

    def predict(self, users_list, k=5) -> np.array:
        """
        Computes first k recomendations for users in list
        Parameters
        ----------
        users_list : np.array
            An array of users uids who needs recommendations
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        recommendations : np.array
            Ordered array with recommended users uids
        """
        recomendations = []
        for user in tqdm(users_list):
            baseline_recommendation = self._find_nearest_neighbors(user=user, k=k, method=self.method)
            recomendations.append(baseline_recommendation)
        return recomendations


@attr.s()
class KNNModel(BaseModel):
    logging.basicConfig(filename="UserBased.log", level=logging.INFO)
    logger = logging.getLogger('UserBased')

    def fit(self):
        self.model = linear_kernel(self.train_csr, self.train_csr)

    def rec_calc(self, idx) -> np.array:
        recommendations = self.model[idx]
        return recommendations

    def predict(self, users_list, k=5) -> np.array:
        """
        Computes first k recomendations for users in list
        Parameters
        ----------
        users_list : np.array
            An array of users uids who needs recommendations
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        recommendations : np.array
            Ordered array with recommended users uids
        """
        if self.ind2uid.isin(users_list).any():
            valid_idx = self.uid2ind[users_list]
            cold_start = np.argwhere(np.isnan(valid_idx)).flatten()
            valid_idx = valid_idx.dropna().astype(int).values
            model_recs = self.rec_calc(valid_idx)
            users_list = self.ind2uid[valid_idx]
            for i, (ind, uid) in enumerate(zip(valid_idx, users_list)):
                model_recs[i, ind] = -1
                if uid in self.neighbors:
                    friends_uids = self.neighbors[uid]
                    friends_idx = self.uid2ind.loc[friends_uids].dropna().astype(int).values
                    model_recs[i, friends_idx] = -1
            sorted_recs = model_recs.argsort()[:, ::-1]
            final_recs = list(map(lambda x: self.ind2uid.loc[x].values, sorted_recs[:, :k]))
            for cold_idx in cold_start:
                baseline_recommendation = self._find_popular[:k]
                final_recs = np.insert(final_recs, cold_idx, baseline_recommendation, axis=0)
            return final_recs
        else:
            baseline_recommendation = self._find_popular[:k]
            return np.array([baseline_recommendation] * len(users_list))


@attr.s()
class CollaborativeFilteringModel(KNNModel):
    logging.basicConfig(filename="CollaborativeFilteringModel.log", level=logging.INFO)
    logger = logging.getLogger('CollaborativeFilteringModel')

    def __attrs_post_init__(self):
        self._load_data(name='_idf')
        self._set_uids2indiced_map()

    def fit(self):
        self.model = TruncatedSVD(random_state=self.random_state)
        self.model.fit(self.train_csr)
        self.uid1_repres = self.model.transform(self.train_csr)
        self.uid2_repres = self.model.components_

    def rec_calc(self, idx) -> np.array:
        recommendations = np.dot(self.uid1_repres[idx, :], self.uid2_repres[:, :])
        return recommendations
