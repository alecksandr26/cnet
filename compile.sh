
BUILD_DIR=build-debug
if [ "$1" = "release" ]; then
    BUILD_DIR=build-release
fi

if [ "$1" = "clear" ]; then
    rm -rf build-debug
    rm -rf build-release
    exit 0
fi

TYPE=Debug
if [ "$1" = "release" ]; then
    TYPE=Release
fi

cmake -DCMAKE_BUILD_TYPE=$TYPE -B $BUILD_DIR
# cmake -DCMAKE_BUILD_TYPE=$TYPE -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -B $BUILD_DIR
cmake --build $BUILD_DIR

if [ "$2" = "test" ]; then
    ctest $BUILD_DIR/CTestTestfile.cmake --test-dir $BUILD_DIR/ --verbose 2>&1
fi

