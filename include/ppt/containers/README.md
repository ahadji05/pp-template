## Containers are generic data-structures that serve as building-blocks for the development of portable applications. Within the context of object-oriented programming, they represent the main entities of the application we are developing.

Containers, such as the example container `X` below, are generic data-structures (e.g. `class`, or `struct`) with respect to a template parameter called `MemSpace`, which determines the location of the data.

```cpp
template<class MemSpace>
class X {
  public:
    X();                      // default constructor
    X(std::vector<d_type>& ); // construct from STL container vector
    X(std::ifstream& );       // construct from file
    X(d_type* , size_t );     // construct from raw host-pointer
    
    X(const X&);              // copy constructor
    X &operator=(const X&);   // copy assignment operator
    
    ~X();                     // destructor
    
    //..other public methods
    
  private:
    d_type *_p;               // data     (location -> MemSpace)
    size_t _n;                // metadata (location -> Host)
};
```

The `MemSpace` is expected to be a concrete class that provides as `static methods` the memory management operations:
- `MemSpace::allocate`
- `MemSpace::release`
- `MemSpace::copy`
- `MemSpace::copyFromHost`
- `MemSpace::copyToHost`

These are necessary for proper management of the resources through well defined Construction, Cleanup, and Copy operations.

- Construction: The construction of objects is done through one of the available constructors based on the input parameter. Each constructor initializes the data explicitly on host and then copies them to the generic memory-space `MemSpace`. This behaviour is implemented based on the methods `MemSpace::allocate` and `MemSpace::copyFromHost`. Both of them are provided from the template parameter `MemSpace`.

- Clean-up: The clean-up is done through the destructor which is implemented based on `MemSpace::release`.

- Copy: Objects are copied based on the copy constructor and the copy-assignment operator. For the implementation of these operations the `MemSpace::copy` is used, which performs deep-copy within the memory-space.
