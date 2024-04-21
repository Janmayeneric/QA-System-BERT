

<!-- distillBert.py -->
## About the Project

This is the project code to develop a foundational Question and Answer (Q&A) system for further uses. The primary focus is to develop the Q&A system that can process the basic user queries.

The first model employs [DistilBERT](https://arxiv.org/abs/1910.01108), imported from [Hugging Face](https://huggingface.co/docs/transformers/model_doc/distilbert). Functioning as an 'Extractive' AI, it extracts answers from provided contexts. 
## Environment For Running

DistilBERT heavily relies on the [transformers](https://huggingface.co/docs/transformers/index) library. Hugging Face recommends an [optimal environment](https://huggingface.co/docs/transformers/installation) for running Transformers offline.

As the developer of this code, I've listed the specific environment configurations required to replicate my setup and ensure the code runs smoothly.
* python 3.9.18
* pytorch 1.13.1
* pytorch-cuda 11.7
* transformers 4.39.2
* dataset 2.18

I utilize a [conda](https://www.anaconda.com/) environment to set up PyTorch GPU, which is essential for facilitating my project. This configuration has proven particularly beneficial for me, especially considering the challenges of configuring such environments on a Windows 10 system.

```sh
conda install pytorch==1.31.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Hope it works.

### ON Unix Cluster ([JHPCE](https://jhpce.jhu.edu/) approach)
I also implemented the bash command to establish a stable conda environment for the code to run. Check from this [directory](https://github.com/Janmayeneric/QA-System-BERT/tree/main/bash).

## How to Initiate Model?

The tutorial is under the directory [tutorial](https://github.com/Janmayeneric/QA-System-BERT/tree/main/tutorial) for you to know how the code in distilBERT.py means.

## Procedure
Again, I recommend you to looks through the [Bash Description](https://github.com/Janmayeneric/QA-System-BERT/tree/main/bash) before this section

First of all, I assume you are in the directory of ~/QA-System-BERT

### First Run, Initialization Example
You need to give the permission to /QA-System-BERT/bash/install_conda_environment.sh
```sh
chmod +x ./bash/install_conda_environment.sh
```

Then run it to install conda Environment
```sh
./bash/install_conda_environment.sh
```

After installing conda environment, you can now give SLURM the job to do, like initialize a model
```sh
sbatch ./bash/initial_model_script.sh
```


