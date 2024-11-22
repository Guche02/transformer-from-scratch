# Transformer from Scratch

This repository implements a Transformer model from scratch, following the original paper "Attention is All You Need" by Vaswani et al. The implementation is split into several components for clarity and modularity.

## Files

- **transformers_scratch.ipynb**: This notebook provides a detailed explanation of the Transformer model as described in the original paper, breaking down its components and functionality.
  
- **model.py**: This is the actual Transformer model without comments, used for training. It includes the architecture, including the encoder-decoder structure, multi-head attention, and feed-forward layers.

- **train.py**: This file is used to train the model. It handles data loading, model initialization, training loops, and logging.

- **dataset.py**: This script is used to load the bilingual English-Italian translation dataset, processing the data to be compatible with the Transformer model.

- **config.py**: Defines the configuration parameters for training. This includes parameters such as batch size, learning rate, and the model's dimensions.

## Usage

** simply clone the repo and run train.py file **
