# ON-POLICY

## 1. Install

### 1.1 instructions

   test on CUDA == 10.1

   

``` Bash
   conda create -n marl
   conda activate marl
   pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
   cd onpolicy
   pip install -e . 
```

### 1.2 hyperparameters

* config.py: contains all hyper-parameters

* default: use GPU, chunk-version recurrent policy and shared policy

* other important hyperparameters:
  - use_centralized_V: Centralized training (MA) or Centralized training (I)
  - use_recurrent_policy: rnn or mlp
  - use_eval: turn on evaluation while training, if True, u need to set "n_eval_rollout_threads"

## 2. StarCraftII

### 2.1 Install StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

*  download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

*  If you want stable id, you can copy the `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

### 2.2 Train StarCraftII

* train_smac.py: all train code

  + Here is an example:

  

``` Bash
  conda activate marl
  cd scripts
  chmod +x train_smac.sh
  ./train_smac.sh
```

  + local results are stored in fold `scripts/results`, if you want to see training curves, login wandb first, see guide [here](https://docs.wandb.com/). Sometimes GPU memory may be leaked, you need to clear it manually.

   

``` Bash
   ./clean_gpu.sh
```

### 2.3 Tips

   Sometimes StarCraftII exits abnormally, and you need to kill the program manually.

   

``` Bash
   ./clean_smac.sh
   ./clean_zombie.sh
```

## 3. Hanabi

  ### 3.1 Hanabi

   The environment code is reproduced from the hanabi open-source environment, but did some minor changes to fit the algorithms. Hanabi is a game for **2-5** players, best described as a type of cooperative solitaire.

### 3.2 Install Hanabi 

   

``` Bash
   pip install cffi
   cd envs/hanabi
   mkdir build & cd build
   cmake ..
   make -j
```

### 3.3 Train Hanabi

   After 3.2, we will see a libpyhanabi.so file in the hanabi subfold, then we can train hanabi using the following code.

   

``` Bash
   conda activate onpolicy
   cd scripts
   chmod +x train_hanabi_forward.sh
   ./train_hanabi_forward.sh
```


## 4. MPE

### 4.1 Install MPE

``` Bash
   # install this package first
   pip install seabon
```

3 Cooperative scenarios in MPE:

* simple_spread: set num_agents=3
* simple_speaker_listener: set num_agents=2, and use --share_policy
* simple_reference: set num_agents=2

### 4.2 Train MPE

   

``` Bash
   conda activate marl
   cd scripts
   chmod +x train_mpe.sh
   ./train_mpe.sh
```

