# Prediction Hubs are Context-Informed Frequent Tokens in LLMs
Code for the article "Prediction Hubs are Context-Informed Frequent Tokens in LLMs" by Beatrix M. G. Nielsen, Iuri Macocco and Marco Baroni, accepted for ACL 2025. 


## Description
We consider hubness in language model representations. We find that:
1. Hubness is not a problem when language models are being used for next-token prediction.
2. Hubness **can** become a problem when:
   * tokens are compared with tokens
   * contexts are compared with contexts
   * contexts and tokens are compared with a different distance than the one the model uses.

Code for running the experiments used in the article is in **experiments**. Code for single plots is also in this folder. 
Code for combined plots and tables is in **article_plots**. 

Plots from **example_plots.py** use synthetic data and can be generated directly with this repo. However, the other plots and tables require either result files with information about hubness or embeddings and/or unembeddings from language models. 


## Installation guide

The outline of the installation is the following:

**1. Create and activate conda environment**

**2. Conda install relevant conda packages**

**3. Pip install relevant pip packages**

In 2. and 3. there might be differences depending on your machine and preferences.

**1. Create and activate conda environment**

Use the commands:
```
conda create -n hubs-freq-tokens python=3.10
conda activate hubs-freq-tokens 
```

**2. Conda install relevant conda packages** 

Install the relevant packages with the following commands:
```
conda install docopt h5py matplotlib pandas scikit-learn scipy tqdm
```
If you are on a CPU only machine continue with:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
If you are on a GPU machine use instead with the relevant cuda version:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
```
Finish by installing the transformers package 
```
conda install conda-forge::transformers
```


**3. Pip install relevant pip packages** 

For GPU machines also install the following:
```
pip install 'accelerate>=0.26.0'

pip install bitsandbytes==0.44.1
```

## Datasets




WIP: More description coming 
