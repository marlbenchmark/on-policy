# MAPPO

## New Update！！！

We support SMAC V2 now～

Chao Yu*, Akash Velu*, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. 

This repository implements MAPPO, a multi-agent variant of PPO. The implementation in this repositorory is used in the paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (https://arxiv.org/abs/2103.01955). This repository is heavily based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail. We also make the off-policy repo public, please feel free to try that. [off-policy link](https://github.com/marlbenchmark/off-policy)

<font color="red"> All hyperparameters and training curves are reported in appendix, we would strongly suggest to double check the important factors before runing the code, such as the rollout threads, episode length, ppo epoch, mini-batches, clip term and so on. <font color='red'>Besides, we have updated the newest results on google football testbed and suggestions about the episode length and parameter-sharing in appendix, welcome to check that. </font>

<font color="red"> We have recently noticed that a lot of papers do not reproduce the mappo results correctly, probably due to the rough hyper-parameters description. We have updated training scripts for each map or scenario in /train/train_xxx_scripts/*.sh. Feel free to try that.</font>


## Environments supported:

- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)
- [Hanabi](https://github.com/deepmind/hanabi-learning-environment)
- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)
- [Google Research Football (GRF)](https://github.com/google-research/football)
- - [StarCraftII (SMAC) v2](https://github.com/oxwhirl/smacv2)

## 1. Usage
**WARNING: by default all experiments assume a shared policy by all agents i.e. there is one neural network shared by all agents**

All core code is located within the onpolicy folder. The algorithms/ subfolder contains algorithm-specific code
for MAPPO. 

* The envs/ subfolder contains environment wrapper implementations for the MPEs, SMAC, and Hanabi. 

* Code to perform training rollouts and policy updates are contained within the runner/ folder - there is a runner for 
each environment. 

* Executable scripts for training with default hyperparameters can be found in the scripts/ folder. The files are named
in the following manner: train_algo_environment.sh. Within each file, the map name (in the case of SMAC and the MPEs) can be altered. 
* Python training scripts for each environment can be found in the scripts/train/ folder. 

* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 


## 2. Installation

 Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/). We remark that this repo. does not depend on a specific CUDA version, feel free to use any CUDA version suitable on your own computer.

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install on-policy package
cd on-policy
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 2.1 StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

* download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

For SMAC v2, please refer to https://github.com/oxwhirl/smacv2.git. Make sure you have the `32x32_flat.SC2Map` map file in your `SMAC_Maps` folder.

### 2.2 Hanabi
Environment code for Hanabi is developed from the open-source environment code, but has been slightly modified to fit the algorithms used here.  
To install, execute the following:
``` Bash
pip install cffi
cd envs/hanabi
mkdir build & cd build
cmake ..
make -j
```
Here are all hanabi [models](https://drive.google.com/drive/folders/1RIcP_rG9NY9UzaWfFsIncDcjASk5h4Nx?usp=sharing).

### 2.3 MPE

``` Bash
# install this package first
pip install seaborn
```

There are 3 Cooperative scenarios in MPE:

* simple_spread
* simple_speaker_listener, which is 'Comm' scenario in paper
* simple_reference

### 2.4 GRF

Please see the [football](https://github.com/google-research/football/blob/master/README.md) repository to install the football environment.

## 3.Train
Here we use train_mpe.sh as an example:
```
cd onpolicy/scripts
chmod +x ./train_mpe.sh
./train_mpe.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

We additionally provide `./eval_hanabi_forward.sh` for evaluating the hanabi score over 100k trials. 

## 4. Publication

If you find this repository useful, please cite our [paper](https://arxiv.org/abs/2103.01955):
```
@inproceedings{
yu2022the,
title={The Surprising Effectiveness of {PPO} in Cooperative Multi-Agent Games},
author={Chao Yu and Akash Velu and Eugene Vinitsky and Jiaxuan Gao and Yu Wang and Alexandre Bayen and Yi Wu},
booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2022}
}
```

