# Reproducability-Study
>📋  A template README.md for code accompanying a Machine Learning paper
# Interpretable Clustering on Dynamic Graphs with Recurrent Graph Neural Networks

This repository is the official implementation of Interpretable Clustering on Dynamic Graphs with Recurrent Graph Neural Networks(https://arxiv.org/abs/2012.08740). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials
## Requirements

To install requirements:

```setup
run python
pip install tensorflow-gpu
pip install tensoflow_addons
pip install numpy
pip install spektral
```

To run non model main:

```setup
run python 3.6
pip install dgl-cu101
pip install dynamicgem
pip install keras==2.2.4
pip install torch
pip install --user scipy==1.4.1
pip install sklearn
```

These are seperate because dynamicgem has a lot of specific dependencies that make it incompatible with the original environment

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
## Training

To train the model(s) in the paper, run:

```train
main.ipynb
```

To train models which don't need to be trained in the paper, run:
```train
non model main.ipynb
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.
## Evaluation

To evaluate the model(s) in the paper, run:

```train
main.ipynb
```

To evaluate models which don't need to be trained in the paper, run:
```train
non model main.ipynb
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
## Pre-trained Models:

Pre-trained Models are not available yet. Models take around 5 minutes to train.

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
## Results

Our model achieves the following performance on :

### DBPL3(https://paperswithcode.com/paper/interpretable-clustering-on-dynamic-graphs)

| Model name         |    Accuracy     |       AUC       |       F1        |
| ------------------ |---------------- | --------------- | --------------- |
| Graphsage          |     77.56%      |      86.46%     |     69.77%      |
| GCN                |     78.34%      |      89.12%     |     69.45%      |
| GAT                |     78.17%      |      87.76%     |     68.81%      |
| Dynaernn           |     45.69%      |      51.57%     |     52.34%      |
| Spectral           |     76.22%      |      50.26%     |     66.63%      |
| GCNLSTM            |     77.48%      |      86.5%      |     70.56%      |
| RNNGCN             |     77.83%      |      88.28%     |     69.26%      |
| TRNNGCN            |     77.84%      |      87.39%     |     69.51%      |

### DBPL5(https://paperswithcode.com/paper/interpretable-clustering-on-dynamic-graphs)

| Model name         |    Accuracy     |       AUC       |       F1        |
| ------------------ |---------------- | --------------- | --------------- |
| Graphsage          |     66.5%       |      80.49%     |     58.9%       |
| GCN                |     68.5%       |      87.67%     |     56.31%      |
| GAT                |     68.74%      |      86.97%     |     56.67%      |
| Dynaernn           |     37.36%      |      51.06%     |     41.63%      |
| Spectral           |     67.3%       |      54.16%     |     50%         |
| GCNLSTM            |     67.68%      |      84.57%     |     57.66%      |
| RNNGCN             |     68.55%      |      85.99%     |     57.85%      |
| TRNNGCN            |     68.65%      |      85.85%     |     57.58%      |

### Reddit(https://paperswithcode.com/paper/interpretable-clustering-on-dynamic-graphs)

| Model name         |    Accuracy     |       AUC       |       F1        |
| ------------------ |---------------- | --------------- | --------------- |
| Graphsage          |     28.8%       |      56.37%     |     16.38%      |
| GCN                |     29.24%      |      55.93%     |     18.75%      |
| GAT                |     31.85%      |      55.92%     |     15.54%      |
| Dynaernn           |     29.16%      |      52.44%     |     28.98%      |
| Spectral           |     32.02%      |      50.14%     |     15.93%      |
| GCNLSTM            |     31.23%      |      56.7%      |     20.93%      |
| RNNGCN             |     31.85%      |      55.92%     |     15.54%      |
| TRNNGCN            |     30.96%      |      56.18%     |     17.57%      |

### Brain(https://paperswithcode.com/paper/interpretable-clustering-on-dynamic-graphs)

| Model name         |    Accuracy     |       AUC       |       F1        |
| ------------------ |---------------- | --------------- | --------------- |
| Graphsage          |     64.93%      |      91.29%     |     91.29%      |
| GCN                |     21.12%      |      67.62%     |     12.56%      |
| GAT                |     39.81%      |      82.6%      |     33.18%      |
| Dynaernn           |     26.28%      |      58.61%     |     26.01%      |
| Spectral           |     36.36%      |      64.18%     |     36.68%      |
| GCNLSTM            |     41.52%      |      85.1%      |     40.1%       |
| RNNGCN             |     30.04%      |      76%        |     24.66%      |
| TRNNGCN            |     21.94%      |      66.42%     |     15.58%      |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
