#!/bin/sh

image=$1

sudo docker run \
    -v $(pwd)/opt_ml/input:/opt/ml/input \
    -v $(pwd)/opt_ml/model:/opt/ml/model \
    -v $(pwd)/opt_ml/output:/opt/ml/output \
    -p 8080:8080 --rm ${image} serve
