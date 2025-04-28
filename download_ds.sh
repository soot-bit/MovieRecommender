mkdir -p Data

echo "📥 Downloading..."
axel -n 2 -o Data/ https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
axel -n 20 -o Data/ https://files.grouplens.org/datasets/movielens/ml-latest.zip

echo "Extracting... 📂"
find Data/ -name "*.zip" -exec sh -c 'unzip -q "$1" -d Data/ && rm "$1"' sh {} \;

echo "༄🧹 ..."
echo "👍"
