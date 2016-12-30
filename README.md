# Binding by Reconstruction Clustering in TensorFlow
A TensorFlow implementation of the perceptual grouping framework as described in ["Binding by Reconstruction Clustering"](http://arxiv.org/abs/1511.06418) with several extensions. Additional resources can be found [here](https://github.com/Qwlouse/Binding) 

## Extensions

* Support for multi-channel images
* Support for real-valued images
* Support for noise adaptation to the data distribution

## Dependencies and Setup

* tensorflow >= 0.12
* numpy >= 1.8
* sacred >= 0.6.7
* pymongo
* h5py
* sklearn

Optional (debugging purposes)

* scipy with Pillow back-end installed 

## Code
Non-TensorFlow RC code to run the reconstruction clustering algorithm has been adapted from https://github.com/Qwlouse/Binding with permission.
