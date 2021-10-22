# Reproducability-Study
>📋  A template README.md for code accompanying a Machine Learning paper
# Dynamic Graph Clustering with Hybrid RNN-GCNArchitecture

This repository is the official implementation of Interpretable Clustering on Dynamic Graphs with Recurrent Graph Neural Networks(https://arxiv.org/abs/2012.08740). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials
## Requirements

To install requirements:

```setup
pip install tensorflow-gpu
pip install tensoflow_addons
pip install numpy
pip install spektral
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
## Training

To train the model(s) in the paper, run:

```train
main.ipynb
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.
## Evaluation

To evaluate my model on ImageNet, run:

```eval
main.ipynb
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
## Pre-trained Models:

Pre-trained Models are not available yet.

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
## Results

Our model achieves the following performance on :

### DBPL3(https://paperswithcode.com/paper/interpretable-clustering-on-dynamic-graphs)

| Model name         |    Accuracy     |       F1       |
| ------------------ |---------------- | -------------- |
| Graphsage          |     64.9%       |      51.0%     |
| GCN                |     66.5%       |      48.2%     |
| GAT                |     62.3%       |      53.9%     |
| RNNGCN             |     65.7%       |      55.4%     |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
