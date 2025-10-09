

# RBML-Diff

This repository is the official implementation of [RBML-Diff: Diffusion Model with Region-Boundary Mutual Learning for Polyp Seg-
mentation](). 

Our implementation is based on the denoising diffusion repository from <a href="https://github.com/M0ck4ry/RBML-Diff">lucidrains</a>

And we provide our pretrained weight and inference result in release.

## Requirements
- python == 3.9
- cuda == 11.3

To install requirements:

```setup
pip install -r requirements.txt
```

[//]: # (>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)


## Training

To train the model(s) in the paper, run  [train.py](train.py) :





## Evaluation
To test a model, run [sample.py](sample.py) 

