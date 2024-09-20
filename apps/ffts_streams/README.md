## Example that demonstrates how to combine streams and cuda ffts to overlap computation with data-transfers.

### To configure building using CMake do:
```sh
mkdir build
cd build
cmake .. -Dppt_ROOT=/path/to/ppt/installation
```

### Then, compile and run the executable: ( first param -> signal-length, second param -> repetitions )
```sh
make
./main_exe 100000 10
```

### Expected output
```
max_diff1: 0
max_diff2: 0
```