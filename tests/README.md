### Tests are applied to the level of generic *containers*, *routines*, and *algorithms*.

In file `/include/ppt/types.hpp`, the execution and memory spaces are specified by conditional code compilation:
- The execution-space that we test against is defined with the alias `exec_space` ,
- The memory-space is resolved directly from the execution-space with the alias `memo_space` . 

`exec_space` and `memo_space` resolve the back-end that we tests against. All unit-tests are developed based on these two aliases so they are generic.
Each time we want to test against a different back-end the code needs to be recompiled accordingly.

The unit-tests are organized in the directories `/containers/`, `/routines/`, and `/algorithms/` accordingly. 
This setup is handled by the `CMake` build environment that is already in place.
