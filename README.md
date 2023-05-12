## Template for Performance-Portable code development, targeting HOST-DEVICE computer architectures.

### The template comes with:
- Organized directories-hierarchy that distinguishes source-code (host and device), and header files.
- Doxygen for automated documentation based on the directories-hierarchy.
- GTest (Google-test) environment for unit-testing based on the directories-hierarchy.
- CMake build-environment that manages:
    - build and installation of the source code as a static library
    - build of application(s) that are developed on top of the source-code
    - build of unit-tests
    - build of documentation

To build source code and run the application, it is recommended to use the CMake build-environment that is provided, following the instructions in `BUILD.md`. The minimum CMake-version required is **3.18**.

The provided build-environment compiles and installs the source-code as a *static library* in the specified `CMAKE_INSTALL_PREFIX` path (see `BUILD.md`). If no path is provided, the CMAKE default-path is used, which for linux operating-systems is: `/usr/local/`.

### Apps:

In the `/apps/` directory a *Proof-Of-Concept* application is developed based on the template, which simulates 2D wave propagation based on finite-difference modelling.