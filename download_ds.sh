#!/bin/bash

mkdir -p Data

echo "📥"
curl -L https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -o Data/ml-latest-small.zip
curl -L https://files.grouplens.org/datasets/movielens/ml-latest.zip -o Data/ml-latest.zip


echo "Extracting... 📂"
unzip Data/*.zip -d Data


echo "༄🧹"
rm Data/*.zip

echo "👍"
