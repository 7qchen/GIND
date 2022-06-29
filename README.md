<h2 align="center">GIND: Optimization-Induced Graph Implicit Nonlinear Diffusion</h2>

This repo contains the implementation of the *<ins>G</ins>raph <ins>I</ins>mplicit <ins>N</ins>onlinear <ins>D</ins>iffusion* model, as described in our paper: 

Qi Chen, Yifei Wang, Yisen Wang, Jiansheng Yang, Zhouchen Lin: **[Optimization-Induced Graph Implicit Nonlinear Diffusion]** *(ICML 2022)*

*GIND* is implemented in [PyTorch](https://pytorch.org/) and utilizes the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) (PyG) library.

## Requirements

* Install [**PyTorch >= 1.7.0**](https://pytorch.org/get-started/locally/)
* Install [**PyTorch Geometric >= 1.7.0**](https://github.com/rusty1s/pytorch_geometric#installation)

We use [**Hydra**](https://hydra.cc/) to manage hyperparameter configurations.

## Project Structure

* **`model/`** contains the model architecture of *GIND*
* **`lib/`** contains helpful functions used in our experiments
* **`node_classification/`** includes experiments to evaluate *GIND* on node classification tasks
* **`graph_classification/`** includes experiments to evaluate *GIND* on graph classification tasks

## Cite

If you find this repo useful, please cite:

```
@inproceedings{chen2022optimization,
  title={Optimization-Induced Graph Implicit Nonlinear Diffusion},
  author={Chen, Qi and Wang, Yifei and Wang, Yisen and Yang, Jiansheng and Lin, Zhouchen},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022},
}
```
