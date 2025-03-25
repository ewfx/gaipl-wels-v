#!/bin/bash

# Update package lists
sudo apt update -y

# Install Python (change version if needed)
sudo apt install -y python3 python3-pip

# Verify installation
python3 --version
pip3 --version

echo "Python installation complete."
