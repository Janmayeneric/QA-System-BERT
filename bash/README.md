# Introduction to Bash Command

In order to run on the cluster, you need to establish a Conda environment on the cluster and then use it with SLURM to handle the rest. In these instructions, I have put together the necessary bash commands for you to initialize your own Conda environment for running. Additionally, I have included the bash commands that could work with SLURM on the JHPCE cluster. However, this is just a basic setup. For further information, such as how to allocate different combinations of computational resources, please refer to the [SLURM instructions](https://jhpce.jhu.edu/slurm/environment/). My aim is to ensure successful execution and verify its functionality.

## Procedure

**REMEMBER!!!!!!!!!!!!!!!!!** 

**YOU ALWAYS SHOULD INSTALL A CONDA ENVIRONMET(install_conda_environment.sh) ON NEW COMPUTER NODE**

Other bash is for inital the model and run the model under the conda environment, so you have to install a conda environment when you run that on new machine.


### Install Conda environment for BERT(intall_conda_environment.sh)

In order to train or use the model, a specific environment is required. The following bash command is for setting up that Conda environment. First, you need to run that code on the terminal to assign the necessary permissions to the bash file.
```sh
chmod +x install_conda_environment.sh
```

make sure it is in the correct directory

As the bash file involves
```sh
conda init
```

Different shell has different configuration file, on JHUPCE cluster it is
```sh
source ~/.bashrc
```
But I cannot make sure every cluster is the same, like some use
```sh
source ~/.zshrc 
```

So make sure the shell's configuration as it might cause the conflicts


### Initial model (Initial_model_script.sh)

Most of the codes follow the coding style as the previous one.

**PLEASE CHANGE THE EMAIL RECEIVER on SLURM SETTING**

I do not like to receive tons of the alarm of others' job status T_T
