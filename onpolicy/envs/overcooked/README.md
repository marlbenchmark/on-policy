# Overcooked-AI
<p align="center">
<img src="overcooked_ai_js/images/screenshot.png" width="350">
</p>
<br>

## Introduction

Overcooked-AI is a benchmark environment for fully cooperative multi-agent performance, based on the widly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires taking 3 items and placing them in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

To play the game with some previously trained DRL agents, use [Overcooked-Demo](https://github.com/HumanCompatibleAI/overcooked-demo).

For DRL implementations compatible with the environment and reproducibile results to [our paper](https://bit.ly/2XoYHAm), check out [this repo](https://github.com/HumanCompatibleAI/human_aware_rl).

## Installation

When cloning the repo, make sure you also clone the submodules.

```
git clone --recurse-submodules -j8 git@github.com:HumanCompatibleAI/overcooked_ai.git
```

It is useful to setup a conda environment with Python 3.7:

```
conda create -n overcooked_ai python=3.7
conda activate overcooked_ai
```

To complete the installation, run the following commands:

```
cd overcooked_ai
python setup.py develop
```

In `overcooked_ai_js` there is a javascript implementation of the Overcooked MDP and game visualizer.

To install it, cd into `overcooked_ai_js` and set up the package with `npm install`.

For development, you will also need to install browserify:

```
npm install -g browserify
```

### Verifying Installation

#### Python Code

To verify your python installation, you can try running the following command from the inner `overcooked_ai_py` folder:

```
python run_tests_fast.py
```

or (this can take 5-10 mins):
```
python run_tests_full.py
```

#### Javascript Code

Run tests with `npm run test`. Testing scripts use `jest`, which exposes a `window` object, and so
`npm run build-window` should be run before running modified tests.

`overcooked-window.js` is used for the demo and testing.

## Python Code Structure Overview

`overcooked_ai_py` contains:

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically

`agents/`:
- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic

`run_tests.py`: script to run all tests

## Python Visualizations

One can adapt a version of [this file](https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/overcooked_interactive.py) in order to be able to play games in terminal graphics with custom-defined agents.

## Javascript Visualizations

To run a simple demo that plays a trajectory demonstrating the
transitions in the game (requires having npm installed):

```
$ npm run demo
```
