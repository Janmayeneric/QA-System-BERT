

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



