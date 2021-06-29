# Folder for data

Put the data here ...

The files of ```flickr30k_train/val/test_lowered_caps/images_data.joblib``` are original from the [SGM paper](https://arxiv.org/abs/1910.05134). Here we just performed lowering text and basic cleaning data as there were some duplicates in the original files.

The ```matching.joblib``` files are the corresponding matching index between images and captions.

The ```word2idx.joblib``` files are the number encoded of each word in the vocabulary of captions and images. Since we have two separate word embedding module for visual and textual branches, we split them into distinct files.

The ```init_glove.joblib``` files are the embeded vectors for words in the vocabulary which is initialised by Glove model. These files are used for initialising the word embedding modules in our network.

Put the **Visual Features** folder here. Visual Features Folders can be obtained by running extract_visual_features.py. Or you can download here: Link. This includes the Efficientnet-b5 features extracted for each detected objects and their unions (predicates) in images.

Due to storage limitation for uploading on github, data for training set can be downloaded here: Link
