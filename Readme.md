# Toxic Comment Classifier 

## Objective & Summary

**Classifying online comments based on toxicity (multi-label) using Neural Networks**

* Built 3 Neural Network-based models & stacked them to create a Level 2 Classifier that was used in classifying comments on toxicity 
* Used pre-trained word embeddings (FastText, GloVe) and extended dataset by translating to other languages & back to English 
* Achieved AUC & accuracy of 97% each; created a prototype, a subreddit extension and visualization to help comment moderators

## Description of Files and Folders: 

### Folder 1 : ./models/ <br/>
Folder containing the models used for toxic comment preprocessing and classification <br/>
Files:
- keras_model.h5 - Saved weights and training stats for Model 1 (uses Glove, word2vec and NN)
- toxic_comment_classifier_model_1.h5 - Saved weights and training stats for Model 2 (uses FastText and BiGRU)
- toxic_comment_classifier_model_2.h5 - Saved weights and training stats for Model 3 (uses word2vec and LSTM)
- toxic_L2classifier.h5 - Saved weights and training stats for L2 classifier used for stacking (uses NN)
- toxic_tokenizer_1.pickle - Tokenizer for Model 2
- toxic_tokenizer_2.pickle - Tokenizer for Model 3
- train_scaler.save - Scaling parameter for word vectors in Model 1
- w2vec.model - Word vectors sourced from training dataset, used in Model 1
- glove.twitter.27B.100d.txt 
	- Pretrained 100 dimensional word vectors from twitter consists of 27B tokens, used in Model 1.
	- Note: File not present in the directory, too large to be put in zip file. Please download from http://nlp.stanford.edu/data/glove.twitter.27B.zip


### Folder 2: ./static/

Folder containing css for the flask model

Files:
- ./static/css/bootstrap.min.css - predefined css style sheet from Bootstrap
- ./static/css/signup.css - custom defined css styles

### Folder 3: ./templates/

HTML template used for the website

Files:
index.html - HTML code for the main site. Check HTML file for description of the structure of the webapp

### Folder 4: ./

app.py - Code for creating the webapp. Please check code for description and usage of individual functions

## Installation:

Language: Python: 3.6.4
Packages used:
Below is the list of packages required and the versions used. Ensure that all requirements for the packages below are met

- flask: 0.12.2
- nltk: 3.2.5
- gensim: 3.6.0
- numpy: 1.14.0
- keras: 2.2.4
- sklearn: 0.19.1
- pickle: 4.0
- tensorflow: 1.12.0
- praw: 6.0.0

Files to be downloaded:
glove.twitter.27B.100d.txt : Use glove.twitter.27B.100d.txt. Please download from http://nlp.stanford.edu/data/glove.twitter.27B.zip. <br/>
Copy the downloaded file to ./models/

## Execution:

1. To create the webapp, run app.py. Wait for all packages and models to be loaded. 
2. On successful execution, "Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)" message shall be printed on the console.
3. Navigate to http://127.0.0.1:5000/ . Use Chrome/Firefox/Safari. 
