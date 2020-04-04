"""Code for creating web application for running Comment Toxicity Checker and Subreddit Toxicity Checker"""

import praw  # For reddit api integration
from flask import Flask, render_template, request  # For webapp creation
import string  # for preprocessing in model 1
from nltk.tokenize import TweetTokenizer  # for preprocessing in model 1
from gensim.models.word2vec import Word2Vec  # for sentence vectors in model 1
import numpy as np  # used for combining inputs in model prediction
from keras.models import load_model  # importing keras models (1, 2 and 3)
from nltk.stem import WordNetLemmatizer  # for preprocessing in model 1
import nltk  # for preprocessing in model 1
from sklearn.externals import joblib  # for importing scaler in model 1
import pickle  # for importing tokenizers in model 2 and model 3
from keras.preprocessing import sequence

# creating a web app from Flask
app = Flask(__name__)


# Homepage. Renders index.html with no output parameters passed
@app.route("/")
def index():
    return render_template('index.html', result="", outp="", outp2="")


# Page created when form is submitted. Submission only via POST. Renders index.html
@app.route('/index', methods=['POST'])
def predict():
    """Main function which runs the model based on the type of form filled and renders the appropriate html"""

    # get data from form submitted via POST
    result = request.form

    # default values
    prediction = ""
    outp2 = ""

    # if Comment Toxicity Checker form was used
    if 'sentence' in result.keys():

        # sentence input
        sentence = result['sentence']

        # predict multilabel toxicity vector
        a = np.round(model_L2_predict(sentence), 2)

        # convert to % for display
        prediction = [str(np.round(x * 100, 2)) + "%" for x in list(a)]

    # if Subreddit Toxicity Checker form was used
    elif 'subreddit' in result.keys():

        # default dict and lists created
        outp2 = {}
        comms = []

        # subreddit name as input
        subreddit = result['subreddit']

        # returns toxicity vector and sample comments from reddit_score function for subreddit passed
        toxicity, comms = reddit_score(subreddit)

        # count of comments which are above 50% threshold toxicity level in each vector
        x = np.sum(toxicity > 0.5, axis=0) / toxicity.shape[0]

        # compiling each category into dict
        outp2['Toxic'] = str(np.round(x[0], 2) * 100) + "%"
        outp2['Severely Toxic'] = str(np.round(x[1], 2) * 100) + "%"
        outp2['Obscene'] = str(np.round(x[2], 2) * 100) + "%"
        outp2['Threat'] = str(np.round(x[3], 2) * 100) + "%"
        outp2['Insult'] = str(np.round(x[4], 2) * 100) + "%"
        outp2['Identity-Hate'] = str(np.round(x[5], 2) * 100) + "%"

        # sample comments
        outp2['Eg'] = comms
    return render_template('index.html', result=result, outp=prediction, outp2=outp2)


def loadGloveModel(gloveFile):
    """Function to load pretrained glove vectors into memory

    Args:
        gloveFile: Location of glove vectors txt file
    Returns:
        model: glove vectors as a dictionary
    """

    print("Loading Glove Model")
    f = open(gloveFile, 'r', encoding="utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def initialize():
    """Runs at the time of setting up the webapp. Loads all models, tokenizers and scalers required for prediction
    Creates global variables which can be accessed across functions
    
    Args:
        None
    
    Returns:
        None
    """

    # Requirements for Model 1
    # Files loaded: 
    # 1. Word2Vec model trained on complete dataset
    # 2. Glove model: Pretrained Vectors loaded
    # 3. Scaler: for scaling new input
    # 4. Keras Model: Final model for prediction

    nltk.download('wordnet')
    w2vec_model_loc = "./models/w2vec.model"
    kerasmodel_loc = './models/keras_model.h5'
    gloveFile = "./models/glove.twitter.27B.100d.txt"
    scaler_filename = "./models/train_scaler.save"

    # Dimensions of word vectors and glove vectors used
    global n_dim_glove
    n_dim_glove = 100
    global n_dim
    n_dim = 100

    global gloves
    gloves = loadGloveModel(gloveFile)
    global w2v_func
    w2v_func = Word2Vec.load(w2vec_model_loc)
    global scaler
    scaler = joblib.load(scaler_filename)
    global model
    model = load_model(kerasmodel_loc)

    # Requirements for Model 2
    # 1. Keras Model: Trained BiGRU Keras Model for prediction
    # 2. Tokenizer: for converting the input vectors

    global model_1, model_2, t_1, t_2
    model_1 = load_model('./models/toxic_classifier_model_1.h5')
    with open('./models/toxic_tokenizer_1.pickle', 'rb') as handle:
        t_1 = pickle.load(handle)

    # Requirements for Model 3
    # 1. Keras Model: Trained LSTM Keras Model for prediction
    # 2. Tokenizer: for converting the input vectors

    model_2 = load_model('./models/toxic_classifier_model_2.h5')
    with open('./models/toxic_tokenizer_2.pickle', 'rb') as handle:
        t_2 = pickle.load(handle)

    # Requirements for L2 Classifier
    # 1. Keras Model: Trained NN Keras Model for L2 final prediction
    global model_L2
    model_L2 = load_model('./models/toxic_L2classifier.h5')


##########################################
#### Model 1 preprocessing and prediction
##########################################

def preprocess(sentence):
    """Returns tokens after preprocessing for prediction in Model 1

    Args:
        Sentence: string
    Returns:
        a: List of tokens
    """

    tokenizer = TweetTokenizer()  # define tokenizer
    wordnet_lemmatizer = WordNetLemmatizer()  # define lemmatizer
    sentence = sentence.translate(string.punctuation)  # remove punctuations
    tokens = tokenizer.tokenize(sentence)  # tokenize sentence
    a = []
    for token in tokens:
        a.append(wordnet_lemmatizer.lemmatize(token, pos="v"))  # lemmatize tokens
    return a


def sentence_vector(tokens, n_dim, n_dim_glove, gloves, w2v_func):
    """Returns sentence vector for tokens passed

    Args:
        tokens: List, output of preprocess(sentence)
        n_dim: Integer, dimension of word2vec trained model
        n_dim_glove: Integer, dimension of glove model from loadGloveModel
        gloves: pretrained glove vector defined in preprocess
        w2v_func: trained w2vec function defined in preprocess
    Returns:
        Sentence vector
    """

    vec = np.zeros(n_dim).reshape((1, n_dim))  # create vector to store w2vec output
    vec_glove = np.zeros(n_dim).reshape((1, n_dim_glove))  # create vector to store glove output
    count = 0

    for token in tokens:
        try:
            vec += w2v_func[token].reshape((1, n_dim))  # from word2vec model
            vec_glove += gloves[str(token)]  # from glove model
            count += 1
        except KeyError:
            continue  # if word is not present in either word2vec or in glove
    if count != 0:
        vec /= count
        vec_glove /= count
    vecs = np.concatenate((vec, vec_glove), axis=1)  # create a vector combining both models
    return vecs


def predict_toxicity_mathews(sentence, w2v_func, model, gloves, scaler, n_dim_glove, n_dim):
    """For a sentence given, predict toxicity vector as per Model 1
    
    Args:
        Sentence: String
        w2v_func: word2vector function saved
        model: Keras model
        gloves: gloves pretrained vector dictionary
        scaler: Scaler for scaling final vectors
        n_dim_glove: Dimension of glove vectors
        n_dim: Dimension of word2vec model
    Returns:
        Toxicity vector predicted by Model 1
    """

    tokens = preprocess(sentence)
    senvec = sentence_vector(tokens, n_dim, n_dim_glove, gloves, w2v_func)
    senvec_scaled = scaler.transform(senvec)
    return model.predict(senvec_scaled)


##########################################
#### Model 2 preprocessing and prediction
##########################################

def model_1_preprocess(input, t_1):
    """Preprocessing from tokenizer in Model 2

    Args:
        input: sentence
        t_1: keras tokenizer
    Returns:
        input: tokenized input
    """

    maxlen = 100
    input = t_1.texts_to_sequences(input)
    input = sequence.pad_sequences(input, maxlen=maxlen)
    return input


def model_1_predict(input, t_1):
    """Predicting toxicity vector as per Model 2

    Args:
        input: sentence
        t_1: keras tokenizer
    Returns:
        Toxicity Vector as predicted by Model 2
    """

    input = model_1_preprocess(input, t_1)
    return model_1.predict(input)


##########################################
#### Model 3 preprocessing and prediction
##########################################

def model_2_preprocess(input, t_2):
    """Preprocessing from tokenizer in Model 3

    Args:
        input: sentence
        t_2: keras tokenizer
    Returns:
        input: tokenized input
    """

    maxlen = 200
    input = t_2.texts_to_sequences(input)
    input = sequence.pad_sequences(input, maxlen=maxlen)
    return input


def model_2_predict(input, t_2):
    """Predicting toxicity vector as per Model 3

    Args:
        input: sentence
        t_2: keras tokenizer
    Returns:
        Toxicity Vector as predicted by Model 3
    """

    input = model_2_preprocess(input, t_2)
    return model_2.predict(input)


##########################################
#### Model L2 Classifer
##########################################

def model_L2_predict(sentence):
    """Predicting final toxicity from L2 classifier

    Args:
        input: sentence
    Returns:
        Final Toxicity Vector as predicted by L2 classifier
    """

    m1 = model_1_predict([sentence], t_1)
    m2 = model_2_predict([sentence], t_2)

    m3 = np.round(predict_toxicity_mathews(sentence, w2v_func, model, gloves, scaler, n_dim_glove, n_dim), 2)
    input_18 = np.concatenate([m1, m2, m3], axis=1)
    prediction = np.round(model_L2.predict(input_18), 2)

    return prediction[0]


##########################################
#### Reddit score prediction function
##########################################

def reddit_score(subreddit_name):
    """Return toxicity vector and top comments for the subreddit name passed

    Args:
        subreddit_name: string, name of the subreddit to be evaluated
    Returns:
        toxicity: vector for all 200 comments pulled from 100 new posts
        aa: list of top 10 toxic comments
    """

    # parameter for connecting with api
    client_secret = "i3hnSbOpfBqk7Ud8Ae7WTh464jQ"
    client_id = "XW3pGdNvW-oxZA"
    app_name = "cse6242"
    username = "4throwaway2throwaway"
    password = "mathews12"
    reddit = praw.Reddit(client_id=client_id, \
                         client_secret=client_secret, \
                         user_agent=app_name, \
                         username=username, \
                         password=password)

    # attach to subreddit name passed
    subreddit = reddit.subreddit(subreddit_name)

    # get latest 100 posts from subreddits
    new_posts = subreddit.new(limit=100)

    comments = []

    # extract comments from the new posts. Break when more than 200 comments
    for submission in new_posts:
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            comments.append(comment.body)
        if len(comments) > 200:
            break

    # create toxicity vector for each comment extracted
    toxicity = []
    for comment in comments:
        toxicity.append(model_L2_predict(comment))
    toxicity = np.array(toxicity)

    # source top 10 most toxic comments
    c = {b: u[0] for b, u in zip(comments, toxicity)}
    a = sorted(c.items(), key=lambda kv: kv[1], reverse=True)
    aa = [x[0] for x in a[:10]]

    return toxicity, aa


if __name__ == "__main__":
    # run the initialize file to load all required functions
    initialize()

    # run the web app
    app.run()
