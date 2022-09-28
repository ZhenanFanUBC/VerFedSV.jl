# VerFedSV.jl

`VerFedSV.jl` is a Julia package for fair and efficient contribution valuation for vertical federated learning. 


## Installation
To install, just call
```julia
Pkg.add("https://github.com/ZhenanFanUBC/VerFedSV.jl.git")
```

## Data
Experiments are run on Adult, which is downloaded from [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). You can also mannually download other data sets from the website, and call the function `read_libsvm` to load the data. 

## Examples
* Example for synchronous SGD is contained in `experiment/syn_adult.jl`
* Example for asynchronous SGD is contain in `experiment/asyn_adult.jl`. Note that for asynchronous SGD, we need to start Julia with multiple threads, i.e. 
```juila -t M```
, where `M` is equal to the number of clients. 

