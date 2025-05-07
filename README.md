

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="40" height="40" alt="Python logo" style="vertical-align: middle;"/>
  <span style="font-size: 24px; font-weight: bold; margin: 0 10px;">⟷</span>
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/18/ISO_C%2B%2B_Logo.svg" width="40" height="40" alt="C++ logo" style="vertical-align: middle;"/>
</p>

<p align="center"><strong>Python ↔ C++ Integration</strong><br/>
<em>Powered by <a href="https://pybind11.readthedocs.io/" target="_blank">pybind11</a></em>
</p>



<p align="center">
  <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=25&pause=1500&random=true&width=435&lines=super+fast+training+speeds+;easily+train+in+python+;+++%F0%9F%97%B2+%F0%9F%97%B2+++FLASH+RECOMMENDER+%F0%9F%97%B2+%F0%9F%97%B2;%F0%9F%87%A8%E2%9E%95%E2%9E%95+back-end+%F0%9F%8F%8E%EF%B8%8F" alt="Typing SVG" /></a>
</p>



# Flash Recommender System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Interface-Python⇄C++-yellowgreen" alt="Python-C++ Interface"/>
</p>

<p align="center">
  <strong>C++ Dependencies</strong><br/>
  <img src="https://img.shields.io/badge/Uses-Eigen-blueviolet?logo=c%2B%2B" alt="Eigen"/>
  <img src="https://img.shields.io/badge/Uses-EigenRand-ff69b4?logo=c%2B%2B" alt="EigenRand"/>
  <img src="https://img.shields.io/badge/Uses-xtensor-lightgrey?logo=c%2B%2B" alt="xtensor"/>
  <img src="https://img.shields.io/badge/Builds%20with-CMake-064F8C?logo=cmake" alt="CMake"/>
</p>

<p align="center">
  <strong>Python Dependencies</strong><br/>
  <img src="https://img.shields.io/badge/Uses-NumPy%20%7C%20pandas-informational?logo=python" alt="NumPy and pandas"/>
</p>

<p align="center">
  <a href="https://www.codacy.com/manual/p-ranav/indicators?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=p-ranav/indicators&amp;utm_campaign=Badge_Grade">
    <img src="https://api.codacy.com/project/badge/Grade/93401e73f250407cb32445afec4e3e99" alt="Codacy code quality"/>
  </a>
  <a href="https://github.com/p-ranav/indicators/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow?logo=open-source-initiative" alt="MIT License"/>
  </a>
</p>

----

The Flash Recommender System is a high-performance, scalable recommendation engine powered by cpp backend to the heavy lifting of training Alternating Least Squares (ALS) matrix factorization. It is the fastest implementation training a recommender system with ALS.

---

## Features
- Compiled with Clang LLVM’s optimisations - aggressive inlining, loop unrolling, auto-vectorisation, ... 
- cpp backend uses Eigen, a highly optimised library for matrix algebra — fast Cholesky decomposition and other linear algebra routines.
- OpenMP threading for parallelism, enabling shared-memory multithreading with dynamic workload balancing.


## Installation
**🏗️ CLI UI still under construction**  
this is how to use ... 

your system needs `python3-dev git-lfs cmake unzip axel` and a compiler with LLVM toolchain `sudo [pkg] install clang lldb lld libomp-dev` 
1. **Clone the repository:** 
    ```bash
    git clone --recurse-submodule https://github.com/soot-bit/MovieRecommender.git
    git lfs pull
    pip install -U "pybind11[global]" tqdm optuna
    ```

2. run `$ source build.sh`
2. **Train the model or load trained matrices and vectors for making predictions:**
3. **Make predictions:**



to download the the full datasets yourself Million Users large data set run bypass `git lfs pull`
```bash
$ > ./download_ds.sh
```
example usage
```
time python main.py  --dataset "ml-latest-small"
```
add flag `--flash` for flash training. you might not think anything happened but it did train
add flag `---plot` to confirm if indead it did exectute.


<div align="center">



<!-- 
`pip install flash-rec --upgrade`


```python

import flash_rec as fr

# Blazing-fast recommendations in 3 lines!
model = fr.HyperEngine()
model.train(lightning_mode=True) ## uses cpp backend
recommendations = model.predict(user_id=42, top_k=10)

``` -->


###  Benchmarks

| Operation               | Python 🐢 + NumPy | Flash System ⚡ | Speed-up      |
|------------------------|------------------|-----------------|---------------|
| Matrix Factorization   | -                | -               | ***100.1×***  |
| Recommendation Batch   | -                | -               | **2005.7×**   |

*to be done properly*


![trainloss](results/100ktrain.png)

</div>

## **Recipies**


<div style="text-align: left;">

### Using [`DataIndx`](./src/data.py)

This class provides a very efficient way to index and preprocess user-movie ratings data for training recommendation models ALS.  
It has:
- very clever memory management to avoid duplicating snapTensor  
- Data loading and index mappings for users/movies  
- Feature engineering for movies  
- Train-test splitting with per-user stratification  
- *O(1)* retrieval of user/movie ratings  
- efficient sampling of per-user ratings using sklearn `train_test_split`

**arg:**
- `dataset`: the path to the data("ml-latest" 25 M or "ml-latest-small" 100K)

</div>


**Initialization**

```python
from src.data import DataIndx  

# Load dataset (e.g., ml-latest or ml-latest-small)
data = DataIndx("ml-latest", cache=True)
```
<div style="text-align: left;">
By default, this will:

- Load the dataset from `Data/ml-latest/`
- Create user/movie ID mappings
- Perform a stratified train/test split
- Caches all processed data to disk (cache only the large dataset) 


You can access core components of the class after initialization:

```python
print(data.ratings.head())         # Raw ratings DataFrame
print(len(data.idx_to_user))       # Total number of unique users
print(len(data.idx_to_movie))      # Total number of unique movies

# Get internal tensor for ALS training
tensor = data.snap_tensor
```

The tensor has methods:
- `.add_train(user_idx, movie_idx, rating)`
- `.add_test(user_idx, movie_idx, rating)`
- Use with your ALS trainer directly


 **🏷️ Get Movie Features**

```python
movie_id = 123
features = data.get_features(movie_id)
print(features)

# Output example:
# {
#     'title': 'Toy Story (1995)',
#     'genres': ['Animation', 'Children', 'Comedy'],
#     'tags': ['Pixar', 'funny', 'great animation']
# }
```



The Data directory should look something like..
```
Data/
└── ml-latest/
    ├── ratings.csv
    ├── movies.csv
    └── tags.csv
```
</div>


<div align="center"><details>
<summary><h3>🧠 ALS algorithm</h3></summary>
  

- Update $U \rightarrow V \rightarrow b_i \rightarrow b_j$ iteratively.
 
 **User Vector**
$u_i = \left( \lambda \sum_{j \in \Omega(i)} v_j v_j^T + \tau I \right)^{-1} \left( \lambda \sum_{j \in \Omega(i)} (r_{ij} - b_i - b_j) v_j \right)$

   **Movie Vector**
   $v_j = \left( \lambda \sum_{i \in {\Omega}^{-1}(j)} u_i u_i^T + \tau I \right)^{-1} \left( \lambda \sum_{i \in {\Omega}^{-1}(j)} (r_{ij} - b_i - b_j) u_i \right)$


   **User Bias**
   $b_i = \frac{\lambda \sum_{j \in \Omega(i)} \left(r_{ij} - u_i^T v_j - b_j\right)}{\lambda |\Omega(i)| + \gamma}$

  **Movie Bias**
   $b_j = \frac{\lambda \sum_{i \in {\Omega}^{-1}(j)} \left(r_{ij} - u_i^T v_j - b_i\right)}{\lambda |{\Omega}^{-1}(j)| + \gamma}$


##### **Key Notes:**

- **Interdependence**: Biases $b_i$ and $b_j$ depend on each other, so update them sequentially using the most recent values and split updates
- **Regularization**: $\gamma$ controls the strength of bias regularization. $\tau$ controls strength of latent vec regularisation

</details>




<div align="center">

  [![Made With Love](https://img.shields.io/badge/%F0%9F%92%96-Made_With_Love-ff69b4)](https://github.com/yourusername)
  [![Open Source](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://opensource.org/)

</div>



