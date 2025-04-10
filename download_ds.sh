mkdir -p Data

echo "📥 Downloading..."
axel -n 2 -o Data/ https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
axel -n 20 -o Data/ https://files.grouplens.org/datasets/movielens/ml-latest.zip

echo "Extracting... 📂"
find Data/ -name "*.zip" -exec unzip {} -d Data/ \;

echo "༄🧹 Cleaning up..."
rm Data/*.zip

echo "👍 Done"
