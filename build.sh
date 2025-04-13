rm -rf build/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

BUILD_PATH="$(pwd)/build"

if [[ ":$PYTHONPATH:" != *":$BUILD_PATH:"* ]]; then
    export PYTHONPATH="$BUILD_PATH:$PYTHONPATH"
fi