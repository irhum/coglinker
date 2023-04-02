#!/bin/bash

# Update the package index
sudo apt-get update

# Install texlive-extra-utils and tralics
sudo apt-get install texlive-extra-utils tralics -y

# Clone the s2orc-doc2json repository
git clone https://github.com/irhum/s2orc-doc2json.git

# Move the doc2json directory to the main folder and delete the rest
mv s2orc-doc2json/doc2json .
rm -rf s2orc-doc2json

# Install dependencies
pip install -r requirements.txt