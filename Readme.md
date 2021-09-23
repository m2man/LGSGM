# A Deep Local and Global Scene Graph Matching for Image-Text Retrieval

This is the repository of the **A Deep Local and Global Scene Graph Matching for Image-Text Retrieval** [paper](https://arxiv.org/abs/2106.02400) which is accepted in SOMET2021. This research is inspired by the SGM [paper](https://arxiv.org/abs/1910.05134) and can be considered as an major improvement of it. The comparison can be fully described in our paper.

Our code is mostly based on the SGM original code.

## Update
For those who are interested in MSCOCO Data, I have uploaded the preprocess data and also the original scene graph extracted from images and their captions. You can find them [here](https://drive.google.com/drive/folders/1Q1Msy6kV0pzZ7uxrDjDQW34Ta9CucI4i?usp=sharing). The original data is given by the authors of the SGM model.

## 1. Requirements
Please install packages in the ```requirements.txt```. The project is implemented with python 3.7.9

## 2. Data prepare
Our data (Flickr30k) is original given by the SGM [paper](https://arxiv.org/abs/1910.05134). We only performed same basic cleaning process to remove duplicated data and lowering text. The preprocessed data can be found in the Data folder.

The model also need the visual features which are the embedded vector of objects in images. In this research, we used EfficientNet-b5 to extract the features. You can extract by running ```extract_visual_features.py``` script. We also uploaded our prepared features ([here](https://drive.google.com/drive/folders/1IvlmTZ9wUpOVIr9MzPgWZB5aYTaTD0jn?usp=sharing)). You can download it and place in the **Data folder**.

## 3. Training and Evaluating
You can run the ```main_train.py``` script to perform either training or evaluating the model. Our pretrained model can be found [here](https://drive.google.com/drive/folders/100t_GxbhycwfQO82cz-7Xfkn8_t69_Vz?usp=sharing). Please download it and place in **Report folder**.

## 4. Contact
For any issue or comment, you can directly email me at manh.nguyen5@mail.dcu.ie
