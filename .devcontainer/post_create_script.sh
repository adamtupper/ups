#!/bin/bash

# Switch to project root
cd ..

# Setup virtual environment
virtualenv env -p python3.8
source env/bin/activate

# Install dependencies
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html