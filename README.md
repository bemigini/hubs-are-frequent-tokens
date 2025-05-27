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
We use the 3 datasets made available by [Cheng et al. (2025)](https://openreview.net/attachment?id=0fD3iIBhlV&name=pdf). Each dataset consists of 50K sequences of 20 tokens randomly extracted from Bookcorpus ([Zhu et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf)), Pile10k ([Gao et al., 2020](https://arxiv.org/abs/2201.10005)) and WikiText-103 ([Merity et al., 2017](https://arxiv.org/abs/1609.07843)), respectively.


## Usage
Config files used to point to relevant folders can be found in the config folder.


**model_name_to_path.json** should contain paths to the models. That is if the Pythia models should be loaded from the path "/home/username/pythia", then this should be inserted as the value for the "pythia" key. 


**model_name_to_vocab_folder.json** should contain folders with the unembeddings of the relevant model. The main folder can be set later as the **--vocab-main-folder** argument. The code will expect to find the unembeddings with the file name "unemb.pickle" (except for opt where it is "emb.pickle") in the main folder combined with the relevant folder path from model_name_to_vocab_folder.json. For example for the Pythia model, unembeddings will be loaded from: vocab_main_folder/model_name_to_vocab_folder\['pythia'\]/unemb.pickle. 


To see all options, use:
```
python run.py -h
```
To calculate and save next token probabilities use 
```
run.py save-next-token-probs --output-folder=<file> --vocab-main-folder=<file> --all-emb-folder=<file> [options] 
```

Where **all-emb-folder** is a folder containing files with names "hidden_\{dataset\}\_sane\_\{model_name\}\_reps.pickle" which contains pickled dictionaries with layer numbers as keys and embeddings as values.
Alternatively, embeddings can be saved as h5 files in a folder named "embeddings" at the same level as the configs folder. In this case the files must be named "embeddings\_\{model_name\}_\{dataset\}\_l\{last_layer_number\}_emb.h5" and the **all-emb-folder** argument will be ignored. 

Example of use: 
```
python run.py save-next-token-probs --output-folder=results --vocab-main-folder="../unembeddings" --all-emb-folder=/home/user/representation_pickles --cuda
```


To get examples of hubs, use:
```
run.py get-hub-examples --output-folder=<file> --vocab-main-folder=<file> --copora-folder=<file> [options] 
```
Where copora-folder sets the folder from which to load the datasets. 
Hub examples will be saved to: 

output_folder/hub_examples_top\{top_num_hubs\}\_\{representation_prefix\}\_\{model_name\}\{dataset\}\_\{similarity\}_\{num_neighbours\}.json

Where: 
* top_num_hubs can be set with the --top-num-hubs=\<int\> option (default is 20).
* representation_prefix is "tt" for token to token, "cc" for context to context and "ct" for context to token.
* num_neighbours can be set with the --num-neighbours=\<int\> option (default is 10). 

Example of use:
```
python run.py get-hub-examples --output-folder=results --vocab-main-folder="../unembeddings" --copora-folder=/home/user/copora --cuda
```

Templates for slurm bash scripts can be found next to the run file with names starting with "slurm" and ending with "_ex.sh".  


## Authors and acknowledgment
We thank Santiago Acevedo, Luca Moschella, the members of the COLT group at Universitat Pompeu Fabra and the ARR reviewers for feedback and advice. Beatrix M. G. Nielsen was supported by the Danish Pioneer Centre for AI, DNRF grant number P1. Iuri Macocco and Marco Baroni received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement No.~101019291)  and from the Catalan government (AGAUR grant SGR 2021 00470). This paper reflects the authors’ view only, and the funding agencies are not responsible for any use that may be made of the information it contains.

## License 

See LICENSE.
