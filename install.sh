# /bin/bash
# change source
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

# creat environment
conda create -n ByrdLab --clone base

# activate environment
conda activate ByrdLab

# install packages
conda install jupyter
conda install matplotlib
conda install pylint
conda install networkx

# conda with CPU
# conda install pytorch torchvision torchaudio cpuonly
# conda with cuda
conda install pytorch torchvision torchaudio cudatoolkit=10.1