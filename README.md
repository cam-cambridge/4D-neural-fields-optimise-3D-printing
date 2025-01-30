# 4D Neural Fields Optimise 3D Printing

Official PyTorch codebase for GDIR (the Gradient-Driven Interpolation Regularization) presented in Regularized Interpolation in 4D Neural Fields Enables Optimization of 3D Printed Geometries. 
[\[arXiv\]]()

## Method

Gradient-driven interpolation regularization enables smooth interpolation between observed instances encoded within the same neural field. Notably, this approach introduces inductive biases in regions with sparse supervision by leveraging the network’s own gradients. Specifically, it minimizes the norm of the Jacobian of the output with respect to one of the input dimensions, promoting smoother transitions in underrepresented regions of the input space.

The resulting regularized field can be effectively utilized for various downstream tasks, including video super-resolution and shape interpolation. Additionally, it finds practical applications in industrial settings, such as optimizing geometry as a function of manufacturing process parameters, enhancing the utility and flexibility of this technique.

## Visualizations

As opposed to traditional approaches wherein the field is not regularised, our apporach achieves stable intperpolation between seen geometries even when supervision is extremely sparse.

![Smooth Interpolation](teasers/animated.gif)

## Code Structure

```
.
├── src                       
│   ├── interpol.py           #   the model
│   ├── sine  .py             #   official implementation of SIREN layer
│   ├── utils.py              #   shared utilities
│   └── dataset               #   datasets, data loaders, ...
├── config.yaml               #   the configuration file < activate/deactivate GDIR here >
├── requirements.txt          
└── main.py                   #   entrypoint to launch training locally on your machine

```

### Requirements
* Python 3.8 (or newer)
* PyTorch 2.5
* Other dependencies: pyDOE, numpy, opencv

see requirements.txt

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{margadji4D,
  title={Regularized Interpolation in 4D Neural Fields Enables Optimization of 3D Printed Geometries},
  author={Margadji, Christos and Kuswoyo, Andi and Pattinson, Sebastian},
  journal={arXiv preprint arXiv:},
  year={2025}
}
