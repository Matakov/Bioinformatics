#!/bin/bash 

sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
echo y | sudo apt-get install nvidia-387
sudo apt-mark hold nvidia-387

wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
echo y | sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
echo y | sudo apt-get update
echo y | sudo apt-get install cuda

echo "PATH=usr/local/cuda-8.0/bin:$PATH" >> /etc/environment
source /etc/environment

sudo apt install -y zlib1g-dev libbz2-dev
sudo apt install -y seqan-dev
