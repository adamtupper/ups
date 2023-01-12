#!/bin/bash

# Switch to project root
cd ..

# Setup virtual environment
virtualenv env -p python3.6
source env/bin/activate
python -m ensurepip --upgrade
