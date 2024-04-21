# Remember to chmod +x install_conda_environment.sh before run it

# create a conda environment
module load anaconda

# you can change the name of the environment, it is just a sample name 'qamodel'
conda create --name qamodel python=3.9

# initialize and configure your shell to work with Conda
conda init

# important to execute the conda command in current shell
source ~/.bashrc

conda activate qamodel

# it usually works for the GPU version pytorch
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install transformers
pip install datasets
pip install accelerate -U

# deactivate the3 environment for good practice
conda deactivate