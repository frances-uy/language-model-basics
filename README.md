# UHM ECE 496B Spring 2025 Assignment 1: Basics

This asignment is created from Assignment 1 of [CS336 at Stanford taught in Spring 2024](https://stanford-cs336.github.io/spring2024/). 
For the full description of the original assignment, see the assignment handout at
[cs336_spring2024_assignment1_basics.pdf](./cs336_spring2024_assignment1_basics.pdf)

Check out useful [lectures from CS336 at Stanford](https://github.com/stanford-cs336/spring2024-lectures).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

0. Set up a conda environment and install packages:

``` sh
conda create -n ece496b_basics python=3.10 --yes
conda activate ece496b_basics
pip install -e .'[test]'
```

1. Run unit tests:

``` sh
pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

2. Download the TinyStories data and a subsample of OpenWebText:

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

