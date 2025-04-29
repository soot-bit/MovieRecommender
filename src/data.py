import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import pickle
from rich.console import Console
console = Console()

try:
    import cppEngine
except ImportError:
    import sys
    from pathlib import Path
    build_dir = Path(__file__).parent.parent / "build"  

    if build_dir.exists() and str(build_dir) not in sys.path:
        sys.path.append(str(build_dir))
    try:
        import cppEngine
    except ImportError:
        raise ImportError("Build the cppEngine modules with build.sh")



class DataIndx:
    """
    Data Indexing structure and processor for fast retrieval in training ALS
    with O(1) lookups to efficiently handle sparse matrix sparsity of the data

    Features:
    - smart memory management
    - fast indexing
    - t-t split with per-user stratification & good sampling per user
    - caching for large datasets (e.g. ml-latest)
    """

    CACHE_FILE = "dataindx_cache.pkl"

    def __init__(self, dataset, cache=True):
        self.ds_dir = Path("Data") / dataset
        self.cache_path = self.ds_dir / self.CACHE_FILE

        # init mappings
        self.user_to_idx = None
        self.movie_to_idx = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.movie_to_features = {}
        self._snap_tensor = cppEngine.snapTensor()

        if cache and self.cache_path.exists():
            console.rule(f"[+] Loading cached data_indx")
            self._unserialize()
        else: 
            self._load_csv()
            self._create_mappings()
            self.tt_split()
            self._process_features()
            if cache:
                console.rule("[!] caching")
                self._cache()


    def _cache(self):
        state = {
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'movie_to_idx': self.movie_to_idx,
            'idx_to_movie': self.idx_to_movie,
            'movie_to_features': self.movie_to_features,
            'snap_tensor': self._snap_tensor 
        }
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _unserialize(self):
        with open(self.cache_path, 'rb') as f:
            state = pickle.load(f)
        
        self._snap_tensor = state['snap_tensor']
        
        self.user_to_idx = state['user_to_idx']
        self.idx_to_user = state['idx_to_user']
        self.movie_to_idx = state['movie_to_idx']
        self.idx_to_movie = state['idx_to_movie']
        self.movie_to_features = state.get('movie_to_features', {})

    def _load_csv(self):
        self.ratings = pd.read_csv(
            self.ds_dir / "ratings.csv",
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}
        )
        self.movies = pd.read_csv(
            self.ds_dir / "movies.csv",
            dtype={'movieId': 'int32'}
        )
        self.tags = pd.read_csv(
            self.ds_dir / "tags.csv",
            dtype={'userId': 'int32', 'movieId': 'int32', 'tag': 'string'}
        )

    def _create_mappings(self):
        """bidirectional maps between IDs and indices"""
        unique_users = self.ratings['userId'].unique()
        unique_movies = self.ratings['movieId'].unique()

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = unique_users.tolist()
        self.movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
        self.idx_to_movie = unique_movies.tolist()

    def tt_split(self, test_size=0.2, random_state=42):
        """Train/test split"""
        num_users = len(self.idx_to_user)
        num_movies = len(self.idx_to_movie)
        self._snap_tensor.reshape(num_users, num_movies)

        train_mask = np.zeros(len(self.ratings), dtype=bool)

        for _, group in tqdm(self.ratings.groupby('userId'), desc="Splitting train/test", leave=False):
            idx = group.index
            if len(idx) >= 2:
                _, test_idx = train_test_split(
                    idx,
                    test_size=test_size,
                    random_state=random_state
                )
                train_mask[test_idx] = True

        for idx, row in tqdm(self.ratings.iterrows(), total=len(self.ratings), desc="Populating tensor", leave=False):
            u_idx = self.user_to_idx[row['userId']]
            m_idx = self.movie_to_idx[row['movieId']]
            rating = row['rating']

            if train_mask[idx]:
                self._snap_tensor.add_test(u_idx, m_idx, rating)
            else:
                self._snap_tensor.add_train(u_idx, m_idx, rating)

    @property
    def snap_tensor(self):
        return self._snap_tensor # extra measure to avoid creating duplicate, 

    def _process_features(self):
        """movie features with tags and genres"""
        tag_features = self.tags.groupby('movieId')['tag'].agg(list)
        self.movie_to_features = {}

        for movie_id in tqdm(self.idx_to_movie, desc="Processing features", leave=False):
            movie_data = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            features = {
                'title': movie_data['title'],
                'genres': movie_data['genres'].split('|'),
                'tags': tag_features.get(movie_id, [])
            }
            self.movie_to_features[movie_id] = features

    def get_features(self, movie_id):
        """Get features safely"""
        return self.movie_to_features.get(int(movie_id), {
            'title': 'Unknown',
            'genres': [],
            'tags': []
        })