## Install

### install torch

```
conda create -n <your env> python=3.9
# select specific CUDA version
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### pip install

```
pip install -r requirements.txt
pip install monai[itk]
```

### cuda op

```
cd custom/models/adaptive_clustering_transformer/adaptive_clustering_attention/extensions
pip install -e .

# if error by loading torch, you can choose one command below to try
# pip install -e . --no-build-isolation
# pip install -e . --config-settings editable_mode=compat
# python -m pip install -e . --no-use-pep517

cp build/*/*.so .  # execute this if encountered circle load error
```

## Usage

### config
```
export DATASET_LOCATION=~/add/your/data/here
export EXPERIMENT_LOCATION=~/add/your/exp/result/here
```

### run
```
python train.py -m dataset_folder.dataset.fold=0,1,2,3,4