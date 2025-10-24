#!/bin/bash

if [! -d "data"]; then
  mkdir data
fi

mkdir -p data/prm800k

wget -P data/prm800k/ https://huggingface.co/datasets/tasksource/PRM800K/resolve/main/phase2_train.jsonl
wget -P data/prm800k/ https://huggingface.co/datasets/tasksource/PRM800K/resolve/main/phase2_test.jsonl
