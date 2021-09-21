#!/bin/bash

docker run \
    --rm \
    -v ${PWD}:/w2v \
    -w /w2v \
        doublethinklab/ml-interview-pair-task:latest \
            python train_answer.py
