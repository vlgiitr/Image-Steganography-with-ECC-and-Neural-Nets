#!/bin/bash

cd ./data/
echo "starting dataset download to data folder"
echo "-----------------------------------------------------"

FILE=mnist.zip
if [ -e "$FILE" ]
then
    echo "$FILE already downloaded"
else
    wget https://data.deepai.org/mnist.zip -P ./data/
fi

echo "-----------------------------------------------------"
echo "starting unzip"
echo "-----------------------------------------------------"

if [ -d "preprocessed" ]
then
    echo "already unzipped"
else
    mkdir ./preprocessed 
    pwd
    unzip mnist.zip -d ./preprocessed
fi
