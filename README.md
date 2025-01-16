# 4D Neural Fields Optimise 3D Printing

Official PyTorch codebase for GDIR (the Gradient-Driven Interpolation Regularization) presented in Regularized Interpolation in 4D Neural Fields Enables Optimization of 3D Printed Geometries
[\[arXiv\]]()

## Method

## Visualizations

## Code Structure

```
.                             # the package
├── src                       
│   ├── interpol.py           #   the model
│   ├── utils.py              #   shared utilities
│   └── datasets              #   datasets, data loaders, ...
├── config.yaml               # the configuration file
└── main.py                   # entrypoint to launch GDIR pretraining locally on your machine

```

### Requirements

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{margadji4D,
  title={Regularized Interpolation in 4D Neural Fields Enables Optimization of 3D Printed Geometries},
  author={Margadji, Christos and Kuswoyo, Andi and Pattinson, Sebastian},
  journal={arXiv preprint arXiv:},
  year={2025}
}
