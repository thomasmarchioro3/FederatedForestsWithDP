#!/bin/bash

# NOTE: First download "HuGaDB v2.zip" file from: https://github.com/romanchereshnev/HuGaDB/tree/master
# and copy the file to the root directory of the project

# Create `data/HuGaDB` directory if it doesn't exist
mkdir -p data
mkdir -p data/HuGaDB

# Unzip the downloaded file
unzip HuGaDB\ v2.zip -d data/HuGaDB/

# Remove zip file
rm HuGaDB\ v2.zip

# Create the `metadata` directory if it doesn't exist
mkdir -p metadata

# Cretate the `results` directory if it doesn't exist
mkdir -p results