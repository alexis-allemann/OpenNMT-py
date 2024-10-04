# OpenNMT-py: Open-Source Neural Machine Translation and (Large) Language Models

This is a fork of [`OpenNMT-py`](https://github.com/OpenNMT/OpenNMT-py) library.

This is the code corresponding to the study published in the following paper:

- Allemann A., Atrio À. & Popescu-Belis A. (2024) - Optimizing the Training Schedule of Multilingual NMT using Reinforcement Learning.

## Implementations of baselines and RL-based algorithms

The implementations of the baselines and the RL-based algorithms have been defined in the [`onmt/schedulers`](onmt/schedulers) folder:

- [`onmt/schedulers/Uniform.py`](onmt/schedulers/Uniform.py): Implementation of the baseline with uniform choice between actions.
- [`onmt/schedulers/Proportional.py`](onmt/schedulers/Proportional.py): Implementation of the baseline with random choice of actions but proportional to the amount of data per language.
- [`onmt/schedulers/TSCL.py`](onmt/schedulers/TSCL.py): Implementation of the Teacher-Student Curriculum Learning (TSCL) algorithm.
- [`onmt/schedulers/DQN.py`](onmt/schedulers/DQN.py): Implementation of the Deep Q Network (DQN) algorithm.

## Library adaptations

The following adaptations to the OpenNMT-py library were necessary in order to use the algorithms listed above:

- [`onmt/opts.py`](onmt/opts.py): Script defining the training options for the translation model. Several options have been added to vary the hyperparameters of the curriculum learning algorithms.
- [`train_single.py`](onmt/schedulers/Uniform.py) and [`onmt/inputters`](onmt/inputters): Scripts to manage the loading of data from datasets. It was necessary to manage several inputters in parallel in order to dynamically choose the model’s training languages.
- [`onmt/trainer.py`](onmt/trainer.py): Script that manages the training of a model on a GPU. This script has been adapted to implement curriculum learning. It is in this script that the training is interrupted to adapt the tasks according to the chosen scheduler, and that the states and rewards of the RL are calculated.

## Experiments configuration files

The configuration files for the experiments presented in the paper results are available in the [`config/curriculum_learning`](config/curriculum_learning) folder.