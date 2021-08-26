# GreenLearning.jl
Julia implementation of greenlearning

Original implementation written in Python
- [arXiv](https://arxiv.org/abs/2105.00266)
- [GitHub](https://github.com/NBoulle/greenlearning)
- [Document](https://greenlearning.readthedocs.io/en/latest/)

## Set Up
```bash
git clone git@github.com:yonesuke/GreenLearning.jl.git
cd GreenLearning.jl; git submodule update --init --recursive
```

## Example
- Laplace equation
    Green function and homogeneous solution
    ![laplace_green](examples/laplace/laplace_green_homogeneous.png)
    Loss throuh iteration
    ![laplace_loss](examples/laplace/laplace_loss.png)
