## An Optimized, Easy-to-use, Open-source GPU Solver for Large-scale Inverse Homogenization Problems

![image-20241111133209991](https://s2.loli.net/2024/11/11/jpC3TzYMWXArNIB.png)

This project aims to provide an code framework for efficiently solving the inverse homogenization problems to design microstructure.  

### dependency

* OpenVDB
* CUDA11
* gflags
* glm
* Eigen3

We have packed dependencies into a [conda](https://docs.conda.io/en/latest/miniconda.html) environment (except CUDA  and compilers), you can create it by:

```bash
conda env create -f environment.yml
```

Then you activate it by:

```bash
conda activate homo3d
```



### Compilation

After the dependency is installed, the code can be compiled using cmake:

```shell
mkdir build
cd build
cmake ..
make -j4
```

If the conda environment is activated, `cmake` will automatically checkout the dependencies in this environment.



### Usage

#### command line

* `-reso` : the resolution of the discretized domain, e.g., `-reso 128` defines an $128\times128\times128$ domain. Default value is 128.
* `-obj` : the objective to be optimized, options are `bulk`,`shear`,`npr` and `custom`, which optimizes the bulk modulus, shear modulus, Poisson's ratio and custom objective respectively. Default is `bulk`
* `-init`: the method for initializing the density field, the common and default option is `randc`, which set the initialization via  a set of trigonometric function basis. You can set this option to `manual` to set the initialization from a OpenVDB file.
* `-sym`: symmetry requirement on the structure, only `reflect3`, `reflect6` and `rotate3` are supported. Default is `reflect6`.
* `-vol`: volume ratio for material usage ranging from $(0,1)$, default is `0.3`
* `-E`: Young's  modulus of base material. Default is `1e1` (Recommanded, inappropriate value will cause numerical problem due to poor representation range of Fp16. You can rescale the elastic matrix latter).
* `-mu`: Poisson's ratio of base material. Default is `0.3`
* `-prefix`: output path suffixed with `/` 
* `-in`: variable input determined by other options, e.g., a OpenVDB file path when the argument of `-init` is `manual`.
* `-N`: maximal iteration number, default is `300`.
* `-relthres`: the relative residual tolerance on FEM equation, default is `1e-2`. (The `master` branch may not work well with tolerance smaller than `1e-5`. Usually, the default value is enough to produce a satisfactory result).



#### example

optimizing the bulk modulus :

```shell
./homo3d -reso 128 -obj bulk -init randc -sym reflect6 -vol 0.3 -mu 0.3
```

After the optimization finished, the optimized density field is stored in `<prefix>/rho` in OpenVDB format.

3rd party softwares like Rhino (with grasshopper plugin [Dendro](https://www.food4rhino.com/en/app/dendro)) or Blender may be used to extract the solid part.

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





### Version illustration

If you care more about accuracy rather than performance, please checkout the branch `mix-fp64` and uses a smaller tolerance on  the relative residual of FEM equation:

```bash
./homo3d -reso 128 -vol 0.1 -relthres 1e-6 # set tolerence to 1e-6
```



Other version (branch) such as `mix-fp64fp32` uses a mixed precision scheme and requires less memory.



## Citation

If you are using this project in your academic research, please include the following citation

```
@ARTICLE{Zhang2023-ti,
  title    = "An optimized, easy-to-use, open-source {GPU} solver for
              large-scale inverse homogenization problems",
  author   = "Zhang, Di and Zhai, Xiaoya and Liu, Ligang and Fu, Xiao-Ming",
  journal  = "Structural and Multidisciplinary Optimization",
  volume   =  66,
  pages    =  "Article 207",
  month    =  sep,
  year     =  2023
}

```

