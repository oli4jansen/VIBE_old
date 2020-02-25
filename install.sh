#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv vibe-env

echo "Activating virtual environment"
source $PWD/vibe-env/bin/activate

echo "Installing dependencies"
$PWD/vibe-env/bin/pip install numpy torch torchvision gdown
$PWD/vibe-env/bin/pip install -r requirements.txt

sh prepare_data.sh
sh build_openpose.sh

