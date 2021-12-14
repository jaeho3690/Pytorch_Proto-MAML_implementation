# Pytorch implementation of MAML, Prototypical Network

This repository contains the implementation of MAML and Prototypical Network on miniimagenet dataset. 
The repository is based on neptune logger so you can simply type in your neptune project name and api_token in the `utils.py` file for your use.

# Whats different from all other repositories?
- I thinks its more readable and intuitive
- Neptune logging

# Requirements and installation

The main requirements are: 
- Python 3.8+
- Pytorch 1.8
- torchvision 0.9
- neptune-client 0.13.3

## To install necessary packages
```
pip install 
```
## To Setup for experiment
```
# Download miniimagenet files
sh download.sh

# Setup directory
mkdir episodes
mkdir logging
mkdir results
mkdir checkpoints
```

## Set your neptune api key and neptune project file in utils.py
You must set the NEPTUNE API_TOKEN and NEPTUNE_PROJECT in `utils.py` file for logging. If you do not want logging, you must comment out all logger related codes.

<p align="left"><img width="99%" src="assets/NEPTUNE.png" /></p>

## To run all experiment files
```
sh proto.sh
sh maml.sh
```


## Configurations
There are two main configuration files to understand. `meta_config` contains configurations related to meta learning experiements. We accept changes through argparser for `meta_config`. `model_config` contains all model relted configurations. You can set model specific configurations in the `config` directory.


## Acknowledgment
The construction of dataset was purely mine.
- For the MAML code I took reference from https://github.com/dragen1860/MAML-Pytorch. I took reference of basic network architectures and the update methods
- For prototypical network I took reference from https://github.com/tristandeleu/pytorch-meta. I took reference of the embedding network only. 
- Codes were adapted to suit my dataset and trainer