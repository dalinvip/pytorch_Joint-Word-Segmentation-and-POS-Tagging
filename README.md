# JointPS
A re-implementation of [A Simple and Effective Neural Model for Joint Word Segmentation and POS Tagging](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8351918) based on Pytorch.

# Requirement
	Python  == 3  
	PyTorch == 0.3.1

# Usage  
	modify the config file, detail see the Config directory
	Train:
	(1) sh run_train_p.sh
	(2) python -u main.py --config ./Config/config.cfg --train -p 

# Config
	optimizer: Adam
	lr: 0.001
	dropout: 0.25
	embed_char_dim: 200
	embed_bichar_dim: 200
	rnn_dim: 200
	rnn_hidden_dim: 200
	pos_dim: 100
	oov: avg 
	use_cuda： True or False
	Refer to config.cfg file for more details.

# Network Structure
![](https://i.imgur.com/wIAMutu.png)

# Performance

|  | CTB5 | CTB6 | CTB7 | PKU | NCC |   
| :----------: | ---------- | ---------- | ---------- | ---------- | ---------- |     
| **Model** | **SEG**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**POS** | **SEG**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**POS** | **SEG**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**POS** | **SEG**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**POS** |  **SEG**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**POS** |  
| Our Model (No External Embeddings)  | 97.69&nbsp;&nbsp;&nbsp;&nbsp;94.16 | 95.37&nbsp;&nbsp;&nbsp;&nbsp;90.83 | 95.32&nbsp;&nbsp;&nbsp;&nbsp;90.25 | 95.22&nbsp;&nbsp;&nbsp;&nbsp;92.62 | 93.97&nbsp;&nbsp;&nbsp;&nbsp;89.47 |     
| Our Model (Basic Embeddings)  | 97.93&nbsp;&nbsp;&nbsp;&nbsp;94.44 | 95.78&nbsp;&nbsp;&nbsp;&nbsp;91.79 | 95.77&nbsp;&nbsp;&nbsp;&nbsp;91.12 | 95.82&nbsp;&nbsp;&nbsp;&nbsp;93.42 | 94.52&nbsp;&nbsp;&nbsp;&nbsp;89.82 |      
|**Our Model (Word-context Embeddings)**   | **98.50**&nbsp;&nbsp;&nbsp;&nbsp;**94.95** |**96.36**&nbsp;&nbsp;&nbsp;&nbsp;**92.51** | **96.25**&nbsp;&nbsp;&nbsp;&nbsp;**91.87** | **96.35**&nbsp;&nbsp;&nbsp;&nbsp;**94.14** | **95.30**&nbsp;&nbsp;&nbsp;&nbsp;**90.42** |      




# Cite
	@Article{zhang2018jointposseg,  
	  author    = {Zhang, Meishan and Yu, Nan and Fu, Guohong},  
	  title     = {A Simple and Effective Neural Model for Joint Word Segmentation and POS Tagging},  
	  journal   = {IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP)},  
	  year      = {2018},  
	  volume    = {26},  
	  number    = {9},
	  pages     = {1528--1538},
	  publisher = {IEEE Press},
	}

# Question #
- if you have any question, you can open a issue or email `mason.zms@gmail.com`、`yunan.hlju@gmail.com`、`bamtercelboo@{gmail.com, 163.com}`.

- if you have any good suggestions, you can PR or email me.