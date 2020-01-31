#!/bin/sh
image=$1

mkdir -p $(pwd)/opt_ml/model
mkdir -p $(pwd)/opt_ml/output

rm $(pwd)/opt_ml/model/*
rm $(pwd)/opt_ml/output/*

sudo docker run \
    -v $(pwd)/opt_ml/input:/opt/ml/input \
    -v $(pwd)/opt_ml/model:/opt/ml/model \
    -v $(pwd)/opt_ml/output:/opt/ml/output \
    --rm ${image} train
