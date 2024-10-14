#!/bin/bash

# Download HuGaDB: https://github.com/romanchereshnev/HuGaDB/blob/master/HumanGaitDataBase.zip
wget https://github.com/romanchereshnev/HuGaDB/blob/master/HumanGaitDataBase.zip

# Unzip the downloaded file
unzip HumanGaitDataBase.zip

# Remove zip file
rm HumanGaitDataBase.zip

# Create `data/HuGaDB` directory if it doesn't exist
mkdir -p data/HuGaDB

# Move HumanGaitDataBase to the `data` directory
mv HumanGaitDataBase data/HuGaDB

# Create the `metadata` directory if it doesn't exist
mkdir -p metadata

# Cretate the `results` directory if it doesn't exist
mkdir -p results