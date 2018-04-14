# DeepCollege

<p align="center">
<a href="https://i.imgur.com/OE479F3.png"><img width="650" src="https://i.imgur.com/OE479F3.png" title="source: imgur.com" /></a>
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/deepcollege/deeplearning/master)
[![Discord](https://img.shields.io/discord/102860784329052160.svg)](https://discord.gg/MAMPnmm)
----

### Introduction

This project aims to create a collection of Jupyter notebooks discussing important topics in Deep Learning.

### The tour sequence

- Linear Classifier
- Linear Regression
- Support Vector Machine
- Gradient Descent (No framework)
- Forward props & Back props
- Multilayer Perceptron
- Neural Net (No framework, only numpy)
   - vectors & tensors
   - Different neural net layers (dense, dropout, max-pooling layers)
- Data preprocessing (house price dataset, catvsdog, IMDB)
- K Mean Clustering
- CNN (Pytorch, TF, Keras) -> classify whales photos
- Callbacks (early stopping and etc)
- Tensorboard
- Training model in the cloud
- Sentiment Analysis (binary classification)
- Sentiment analysis (multi-classification)
- Word2vec
- Hyperparamter Tuning
- Transfer Learning
- Generate song lyrics and stories
- Hyperparameter tuning
- Feature engineering
- Linear regression Kaggle competition using the above knowledge
- Bayes Classifier
- Gaussian Mixture Model
- VAE
- Generative Models
- GAN basic
- StarGAN
- CycleGAN
- Reinforcement learning
- Chatbot

### Pre-requesties

- Basic Python Knowledge
- Some Machine Learning
- Basic idea about Deep Learning

### Installation

#### 1. Anaconda

Anaconda is a package management platform Data Scientists that lets you easily manage and install dependencies in 
cross-platform manner. It also ships with *Jupyter Notebook*, which plays a critical role in order to contribute to this
project.

For more detail about why you should use Anaconda? https://www.quora.com/Why-should-I-use-anaconda-instead-of-traditional-Python-distributions-for-data-science

Installation per platform:
- Windows: [Install Python on Windows (Anaconda)](https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444)
- Mac: [Install Python on Mac (Anaconda)](https://medium.com/@GalarnykMichael/install-python-on-mac-anaconda-ccd9f2014072)

#### 2. Docker

Docker is a containerisation technology that enables you to run 
Deep Learning code in environments that are consistent with others. 
Since container are lightweight and intended to be thrown away 
after use, you are technically not installing anything on the 
bare-metal.

1. Tensorflow official image

It includes

- Tensorflow:latest CPU
- Pillow
- h5py
- ipkernel
- jupyter
- matplotlib
- numpy
- pandas
- scipy
- sklearn

1. On Mac or Linux

Install on Mac: https://docs.docker.com/docker-for-mac/install/

Install on Linux: https://docs.docker.com/install/

```sh
$ cd deepcollege/deeplearning
$ docker run -v $(pwd):/notebooks -it -p 8888:8888 gcr.io/tensorflow/tensorflow
```

2. On Windows *Powershell*

Install on Windows: https://docs.docker.com/docker-for-windows/

```sh
$ docker-machine start  # if you are using docker toolbox
$ docker-machine env --shell powershell default | Invoke-Expression
$ cd deepcollege/deeplearning
$ docker run -v /c/Users/<your_user_name>/Desktop/deepcollege/deeplearning:/notebooks -it -p 8888:8888 gcr.io/tensorflow/tensorflow
```

Docker tool-box users tip:
1. When you are mounting volumes, you must convert 
path such as `C://Users/Desktop/code` into `/c/Users/Desktop/code`

### How do you contribute?

1. Join the Discord channel [https://discord.gg/MAMPnmm](https://discord.gg/x6S6ce)
2. Goto _#request-to-join_ channel and post your Github Account name!
3. Once you are granted with access to the project, please create a git branch with your name
4. Complete each topic and challenge in order of number sequence
   - for example: 000-Linear-Classification -> 001-Linear-Regression
5. Reference existing code submissions from contributors or Wiki pages
6. I will post Jupyter Notebooks with sample code or challenges to complete
