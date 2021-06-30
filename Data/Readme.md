# Folder for data

Put the data here ...

Due to storage limitation for uploading on github, data for training set can be downloaded [here](https://drive.google.com/drive/folders/1ZbyeezvaxA20fM7lmxDNL0UnyHX8wLjt?usp=sharing)

The files of ```flickr30k_train/val/test_lowered_caps/images_data.joblib``` are original from the [SGM paper](https://arxiv.org/abs/1910.05134). Here we just performed lowering text and basic cleaning data as there were some duplicates in the original files. Regarding captions triplet data, we preprocessed as those in the SGM paper, which was only keep the relations with full triplet of subject - object - interaction. However, we rearranged the position to subject - interaction - object for each original triplet.

The ```VG-SGG-dicts-with-attri.json``` file is adopted from the repository of scene graph generation Neural Motif model. We used it to encode the words of detected objects in images into index which will be used later in the word embedding module.

The ```matching.joblib``` files are the corresponding matching index between images and captions.

The ```word2idx.joblib``` files are the number encoded of each word in the vocabulary of captions and images. Since we have two separate word embedding module for visual and textual branches, we split them into distinct files.

The ```init_glove.joblib``` files are the embeded vectors for words in the vocabulary which is initialised by Glove model. These files are used for initialising the word embedding modules in our network.

Put the **Visual Features** folder here. Visual Features Folders can be obtained by running ```extract_visual_features.py```. Or you can download [here](https://drive.google.com/drive/folders/1IvlmTZ9wUpOVIr9MzPgWZB5aYTaTD0jn?usp=sharing). This includes the Efficientnet-b5 features extracted for each detected objects and their unions (predicates) in images.