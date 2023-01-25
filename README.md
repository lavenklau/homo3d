## An Optimized, Easy-to-use, Open-source GPU Solver for Large-scale Inverse Homogenization Problems

This project aims to provide an code framework for efficiently solving the inverse homogenization problems to design microstructure.  

### dependency

* OpenVDB
* CUDA11
* gflags
* glm
* Eigen3



### Compilation

After the dependency is installed, the code can be compiled using cmake:

```shell
mkdir build
cd build
cmake ..
make -j4
```



### Usage

#### command line

* `-reso` : the resolution of the discretized domain, e.g., `-reso 128` defines an $128\times128\times128$ domain. Default value is 128.
* `-obj` : the objective to be optimized, options are `bulk`,`shear`,`npr` and `custom`, which optimizes the bulk modulus, shear modulus, Poisson's ratio and custom objective respectively. Default is `bulk`
* `-init`: the method for initializing the density field, the common and default option is `randc`, which set the initialization via  a set of trigonometric function basis. You can set this option to `manual` to set the initialization from a OpenVDB file.
* `-sym`: symmetry requirement on the structure, only `reflect3`, `reflect6` and `rotate3` are supported. Default is `reflect6`.
* `-vol`: volume ratio for material usage ranging from $(0,1)$, default is `0.3`
* `-E`: Young's  modulus of base material. Default is `1e6`
* `-mu`: Poisson's ratio of base material. Default is `0.3`
* `-prefix`: output path suffixed with `/` 
* `-in`: variable input determined by other options, e.g., a OpenVDB file path when the argument of `-init` is `manual`.
* `-N`: maximal iteration number, default is `300`.



#### example

optimizing the bulk modulus :

```shell
./homo3d -reso 128 -obj bulk -init randc -sym reflect6 -vol 0.3 -E 1e6 -mu 0.3
```

After the optimization finished, the optimized density field is stored in `<prefix>/rho` in OpenVDB format.

3rd party softwares like Rhino or Blender may be used to extract the solid part.

The optimized elastic matrix is stored in `<prefix>/C` in binary format, which is an array of 36 float precision numbers.



#### custom objective

To optimizing custom objective, option `-obj custom` should be used and add your objective and optimization routine in `Framework.cu` file, where we have provide few examples：

```cpp
void example_opti_bulk(cfg::HomoConfig config) {
    // ...
}
void example_opti_npr(cfg::HomoConfig config) {
    // ...
}
void example_yours(cfg::HomoConfig config) {
	// Add your routines here....
}
void runCustom(cfg::HomoConfig config) {
	//example_opti_bulk(config);
	//example_opti_npr(config);
	example_yours(cfg::HomoConfig config); // uncomment this line
}
```









