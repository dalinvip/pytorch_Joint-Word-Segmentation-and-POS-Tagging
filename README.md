# JointPS: [A Simple and Effective Neural Model for Joint Word Segmentation and POS Tagging](https://zhangmeishan.github.io/ChineseLexicalProcessing.pdf)
Joint models have shown stronger capabilities for Chinese word segmentation and POS tagging, and have received
great interests in the community of Chinese natural language processing (NLP). In this paper, we follow this line of work, presenting a simple yet effective sequence-to-sequence neural model for the joint task, based on a well-defined transition system, by using long short term memory (LSTM) neural network structures. We conduct experiments on five different datasets. The results demonstrate that our proposed model is highly competitive. By using well-trained character-level embeddings, the proposed neural joint model is able to obtain the best reported performances in the literature.


# Requirement:
	Python  == 3  
	PyTorch == 0.3.1

# Usage  
	modify the config file, detail see the Config directory
	Train:
	(1) sh run_train_p.sh
	(2) python -u main.py --config ./Config/config.cfg --train -p 

# Config


# Network Structure
![](https://i.imgur.com/wIAMutu.png)

# Performance

| 	 | CTB5 | CTB6 | CTB7 |  
| Model | SEG	POS | SEG	POS | SEG	POS |  
| ------------ | ------------ | ------------ | ------------ |  
| BLSTM | 88.64	88.64 | 87.53	88.64 | 88.08	88.64 |  



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

# Schedule
updating.