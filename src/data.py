import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

try:
    import cppEngine
except ImportError:
    import sys
    from pathlib import Path
    build_dir = Path(__file__).parent.parent / "build"  

    if build_dir.exists() and build_dir not in sys.path:
        sys.path.append(str(build_dir))  # Ensure build_dir is added as a string
    try:
        import cppEngine
    except ImportError:
        raise ImportError("Build the cppEngine with build.sh")



class DataIndx:
    """
    Data Indexing structure and processor for fast retrieval in trainig ALS 
    with O(1) look ups to efficient handle sparse matrices sparcity of the data
    
    Does:
    - very clever memory management to avoid duplicating snapTensor
    - Data loading and index mappings for users/movies
    - Feature engineering for movies
    - Train-test splitting with per-user stratification
    - O(1) retrival to user/movie ratings

    arg:
        --- dataset : the dataset directory (ml-latest or ml-latest-small)
    """
    def __init__(self, dataset):
        self.ds_dir = Path("Data") / dataset
        self._snap_tensor = cppEngine.snapTensor()  # Init cpp data structures for used in trainig
        self._load_csv()
        self._create_mappings()
        
    def _load_csv(self):
        self.ratings = pd.read_csv(
                            self.ds_dir / "ratings.csv",
                            dtype={
                                'userId': 'int32', 
                                'movieId': 'int32', 
                                'rating': 'float32'
                                }
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
        """
        Creates bidirectional mappings between user IDs and movie IDs to 
        their corresponding index values. O(1) time complexity and O(n) space

        
        - `user_to_idx` Dic: mapping unique `userId` in the dataset to a 
                             unique integer index (row) of in the snapTensor 
        - `idx_to_user` List: A list of user IDs, where each index corresponds 
                            to a user in the `user_to_idx` mapping.
        - `movie_to_idx` Dic: A dictionary mapping each unique `movieId` to a 
                              unique integer index.
        - `idx_to_movie`List: A list of movie IDs, where each index corresponds
                              to a movie in the `movie_to_idx` mapping.
        """
        unique_users = self.ratings['userId'].unique()
        unique_movies = self.ratings['movieId'].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = unique_users.tolist()
        self.movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
        self.idx_to_movie = unique_movies.tolist()

    def tt_split(self, test_size=0.2, random_state=42):
        num_users = len(self.idx_to_user)
        num_movies = len(self.idx_to_movie)
        
        self.snap_tensor.reshape(num_users, num_movies)

        #  train/test masks
        train_mask = np.zeros(len(self.ratings), dtype=bool)
        for _, group in self.ratings.groupby('userId'):
            idx = group.index
            _, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
            train_mask[test_idx] = True

        # Populate  snap tensor
        for idx, row in self.ratings.iterrows():
            user_idx = self.user_to_idx[row['userId']]
            movie_idx = self.movie_to_idx[row['movieId']]
            rating = row['rating']
            
            if train_mask[idx]:
                self.snap_tensor.add_test(user_idx, movie_idx, rating)
            else:
                self.snap_tensor.add_train(user_idx, movie_idx, rating)
    @property
    def snap_tensor(self):
        return self._snap_tensor

    def _process_features(self):
        """Process features with progress tracking"""
        #  tags
        self.movie_to_features = {} #{movieId -> Action|blah 1|blah 2}
        tag_features = self.tags.groupby('movieId')['tag'].agg(list)
        
        
        for movie_id in tqdm(self.idx_to_movie, desc="Processing features"):
            movie_data = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            features = {
                'title': movie_data['title'],
                'genres': movie_data['genres'].split('|'),
                'tags': tag_features.get(movie_id, [])
            }
            self.movie_to_features[movie_id] = features

    def get_features(self, movie_id):
        """Get features with type checking"""
        return self.movie_to_features.get(int(movie_id), {
            'title': 'Unknown',
            'genres': [],
            'tags': []
        })
