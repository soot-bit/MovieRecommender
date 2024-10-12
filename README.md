
# Recommender System

This repository contains a recommender system implemented using NumPy. The system uses collaborative filtering techniques to generate personalized recommendations based on user-item interaction data.


## Features

- **Collaborative Filtering**: Implements matrix factorization to learn user and item representations.
- **Alternating Least Squares (ALS)**: Efficiently optimizes latent factors through an alternating optimization approach.
- **Bias Handling**: Incorporates user and item biases for improved recommendation accuracy.
- **Feature Support**: Allows for the integration of additional features in the recommendation process.

## Installation

To set up the recommender system, follow these steps:

1. **Clone the repository:**
2. **Train the model:**
3. **Make predictions:**


   
## File Structure

```
MovieRecommender/
│
├── src/                   
│   ├── cpp/                 # C++/CUDA backend 
│   │   ├── recommender.cpp  # Core recommendation logic (training + inference)
│   │   ├── recommender.hpp  # Header file with declarations
│   │   ├── recommender.cu   # CUDA code for acceleration (optional)
│   │   └── Makefile         
│   ├── jax/                 # JAX implementation
│   │   ├── recommender_jax.py 
│   │   └── utils.py         
│   ├── bindings/          
│   │   ├── recommender_pybind.cpp 
│   │   └── setup.py        
│   └── common/              # Shared utilities 
│       ├── data_loader.py  
│       └── config.py        
│
├── main/                  # Main 
│   ├── train.py           
│   ├── recommend.py       
│   ├── benchmark.py       #  C++/CUDA vs JAX
│   └── config.py         
│
├── Data/                # Movie data and model storage
│   ├── 100k.csv         
│   ├── 25M.csv        
│   └── model/            
│
└── README.md             
```

[link for 25 million dataset](http://files.grouplens.org/datasets/movielens/ml-25m.zip)
