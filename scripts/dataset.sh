#!/bin/bash

cd ./data/
echo "starting dataset download to data folder"
echo "-----------------------------------------------------"

FILE=tiny-imagenet-200.zip
if [ -e "$FILE" ]
then
    echo "$FILE already downloaded"
else
    echo "$FILE not found... Starting download"
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
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
    unzip tiny-imagenet-200.zip -d ./preprocessed
fi
