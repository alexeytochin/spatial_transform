# Spatial Transform Networks 

We implement and investigate 
[Spatial Transform Networks](https://arxiv.org/pdf/1506.02025.pdf)
and their extensions with Tensorflow.

## What's implemented
1. Datasets: only [Affine MNIST](https://www.kaggle.com/kmader/affinemnist).
2. Spatial transforms: Affine and Quadratic (see below).
3. INterpolations: Biliniar interpolation. The implementation is based on
 [this blog post](https://towardsdatascience.com/implementing-spatial-transformer-network-stn-in-tensorflow-bf0dc5055cd5)
4. STN-CO, CTN-CX blocks and thier chains with possible shared weights, 
see [paper](https://arxiv.org/pdf/2004.11678.pdf).


## Installation
In as python>=3.7.5 enviroment with tensorflow-gpu>=2.4 install requrements by
```bash
$ conda install --yes --file requirements.txt
```
or
```bash
pip install requirements.txt
```

## Datasets
Dataset affine MNIST can be download 
[here](https://www.kaggle.com/kmader/affinemnist).
Specify the download directory with 
`test.mat`, `validation.mat`, and `training_batches` in 
[`config.py`](config.py).


## Experiments
Jupyter notepads with model training experiments are located in 
[`experiments`](experiments)
directory. 
In order to make advantages of spatial transform more visible on 
Affine MNIST dtaset **we restrict ourself with modle size < 10000 weights**. 

Some experiments are summarized in the table below:

| Name                     |      Description      |  Validation error rate   | Model size | Reference to notebook |
|:-------                  |:---                   |:---                      |:---        |:---                   |
| Convolutional baseline   | Simple conv network   | 0.0161                   | 98694      | [notebook](experiments/Baseline_convolutional_backbone.ipynb)     |
| Basic STN                | Basic STN             | 0.0094                   | 94488      | [notebook](experiments/STN_C0_backbone.ipynb)             |
| Basic STN with CoordConv         | Add coord features to localization network | 0.0082                   | 97656      | [notebook](experiments/Basic_STN_coord_network_0082.ipynb)  |
| STN-CX x 3               | Repeted STN-CX blocks with shared weights        | 0.0139                   | 97520      | [notebook](experiments/STN_CX_backbone.ipynb)               |
| Quadratic transforms     | We try a chine with quadratic transforms         | 0.0274 | 96520 | [notebook](experiments/STN_quadratic_chain_backbone) |


## Interesting observations
1. In the very basic case (see
[notebook](experiments/STN_C0_backbone.ipynb))
the spatial transform maps the digit such that it more or less fits the boundary
of target image. Irrelevant information is mostly left behind. 
See 
2. [CoordConv layer](https://arxiv.org/pdf/1807.03247.pdf) helps to improve the accuracy a bit. 
[notebook](experiments/Basic_STN_coord_network_0082.ipynb) for illustrations.
2. STN_CX_backbone with repeated shared weights overflights see 
[notebook](experiments/STN_CX_backbone.ipynb).
3. We also implement Quadratic transform 

<img src="https://latex.codecogs.com/svg.latex? x_i \rightarrow \tilde x_i = a_i + \sum_j b_{ij} x_j + \sum_{jk} c_{ijk} x_j x_k, \quad i,j,k=1,2, "/>

while the affine transform is

<img src="https://latex.codecogs.com/svg.latex? x_i \rightarrow \tilde x_i = a_i + \sum_j b_{ij} x_j, \quad i,j=1,2. "/>

We try to combine STN-C0 block with such a transform 
in a chain with shared weights in order to get a spatial transform of a
general polynomial form that is follows by a usual convolution NN.
However, this approach did not demonstrate its efficiency in our scope.


## What can be added/improved in future:
1. Add projective transform.
2. Models with composition of spatial transforms. 
3. Optimize bilinear interpolation.
4. Optimize addine and quadratic transforms.