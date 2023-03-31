## Simulation of wave-propagation in a 2D medium with space-dependent velocity. The modelling is done using a second-order in-time, fourth-order in-space, Finite-Difference scheme.

### To configure building using CMake do:
```sh
mkdir build
cd build
cmake ..
```

### Then, compile and run the executable:
```sh
make
./main_exe
```

### For visualizing the output files, use the provided python script:
```sh
python ../visualize.py wavefield.bin velocity_model.bin 701 2001
```