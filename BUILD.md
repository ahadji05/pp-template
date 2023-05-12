## Installing and Using PPT
You can use `PPT` as an installed package in your project. Once `PPT` is installed In your CMakeLists.txt simply use:
```
find_package(ppt REQUIRED)
```
Then for every executable or library in your project that wants to use `PPT`:
```
target_link_libraries(myTarget PPT::ppt)
```

## Configuring CMake
A very basic installation from the top-level directory is done with:
```
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX=/path/to/install/location
make
make install
```
which builds and installs the `PPT` library. The full keyword listing is below.

## PPT Keyword Listing
Options can be enabled by specifying `-DPPT_ENABLE_X`:
- `PPT_ENABLE_TESTS`: BOOL
  - Whether to build tests. Note it requires google's`GTest` to be installed (if GTest is not installed the build environment downloads and installs it automatically).
  - Default: OFF
- `PPT_ENABLE_DOCS`: BOOL
  - Whether to build documentation. Note it requires `Doxygen` to be installed. 
  - Default: OFF
- `PPT_ENABLE_OPENMP`: BOOL
  - Whether to enable OpenMP backend. 
  - Default: OFF
- `PPT_ENABLE_CUDA`: BOOL
  - Whether to enable Cuda backend. 
  - Default: OFF
- `PPT_ENABLE_HIP`: BOOL
  - Whether to enable HIP backend. 
  - Default: OFF

The following options control find_package paths for CMake-based TPLs:
- `GTest_ROOT`: PATH
  - Location of GoogleTest install root.
  - Default: None or the value of the environment variable `GTest_ROOT` if set
- `CUDAToolkit_ROOT`: PATH
  - Location of NVIDIA CUDA Toolkit install root.
  - Default: None or the value of the environment variable `CUDAToolkit_ROOT` if set