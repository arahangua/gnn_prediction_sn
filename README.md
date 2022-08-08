# Using Graph neural networks to predict single neuron firing rate responses in in-vitro networks

This repo stores both computing scripts and data used to generate results in the paper: (link)


# Installation

An exact copy of conda environment is provided 
```
conda_requirements.txt
```

but for most use cases, it would suffice to just run 
```
pip install -r requirements.txt
```

# Description
## folder structure
```
├── conda_requirements.txt
├── data
│   ├── best_params           # folder storing the best parameter that was used in the paper
│   ├── dataset.npy           # final dataset that was used for the prediction tasks
│   ├── example_locs.npy      # (for CCH), this numpy variable stores physical locations of neurons that were used in the notebook example
│   └── example_spktrain.npy  # spike train variable that were used in the notebook example
│   └── extended_dataset.npy
├── FC_scripts
│   ├── assembly_util.py      # utility script for computing Functional connectivity (FC)
│   └── __pycache__
├── LICENSE
├── notebook
│   ├── Computing FCs.ipynb   # notebook tutorial for computing FCs
│   └── prediction_task.ipynb # notebook tutorial for training / testing prediction models 
├── pred_models
│   ├── gnn_torch_models.py   # declares GNN models
│   ├── gnn_torch_utils.py    # util script for computing GNN models
│   ├── non_gnn_models.py     # declares non-GNN models
│   └── __pycache__
├── README.md
├── requirements.txt
└── utils
    ├── pred_utils.py         # utility script in general
    └── __pycache__
```

# Usage 

please check notebook files. ("Computing FCs.ipynb", "prediction_task.ipynb")
