# Natural Language Processing 

This machine learning project makes use of python's libraries such as _nltk_,_re_,_pandas_,_sklearn_,etc to perform text analysis . To add to the complexity of the project an **Artificial Neural Network** has been used as the classifier.

## Part 1: Cleaning the text and preparing the training set

For training the model we use dataset of 1000 restraunt reviews(The dataset is available in the repository). A _.tsv_ file is chosen because normal text can contain commas which mights cause confusion if a _.csv_ file is used. Tab seperates the review and the value 1 or 0 which indicates wheather the review is positive or negative respectively. The _re_ library is used to clean the text and make sure only alphabets remain .  
Then PorterStemmer has been used from the _nltk_ library to remove all the unnecessary words(like prepositions) from the reviews as they don't help us determine wheather the review is positive or negative. this is done for each review which finally leaves us with reviews only consisting of the necessary words in lower case alphabets. 

## Part 2: Tokenisation and creating the bag of words

We now make a matrix of size (1000 * 1500) where the rows are the 1000 reviews while the 1500 most commonly occuring words are the columns. The occurence of these words in positive or negative review can be mapped to predict any future review. This method of breaking up of reviews is known as **tokenisation** and each column of the matrix represents one of the 1500 words and each row represents a review.
**CountVectorizer** provides a simple method for _tokenisation_ and this is available in python's _sklearn_ library. Finaly **Feature Scaling** is performed to standardize the independent features present in the data in a fixed range and even this makes use of the _sklearn_ library.
The dataset also needs to be split into training and testing set as we'll need to check our model on the testing set once we have trained it on the testing set(_A screenshot of what the training set looks like after feature scaling and tokenisation is available in the repository_).

## Part 3: Fitting the Artificial Neural Network into the training set

This is the most crucial part of the project where we use _Keras_ to build a neural network that acts like a classifier for training the dataset. We use 2 models from the _Keras_ library namely _Sequential_ and _Dense_. The _Sequential_ model is used to initialise the neural network whereas we use the _Dense_ model to add hidden layers to our network. For this project we use 2 hidden layers. 
**Optimizers** are algorithms which are used to adjust the attributes of the neural network such as weights and learning rate in order to reduce loss, this project makes use of the _adam_ optimizer. Since the prediction is either positive or negative we use binary cross-entropy as the loss function.
**Activation Function** is associated with each neuron in the network and is used to decide wheather the neuron should be activated or not.For the input layer and the hidden layers it's a good idea to use the _rectifier_ activation function while the output layer makes use of _sigmoid_ activation function.
Finally after compiling the network when we finally fit the training set into the network we have to decide on the **batch size** and **epochs**. for training set of 800 reviews, batch size of 10 and 100 epochs should be efficient. Since the output prediction is a decimal number between 0-1 we consider all the predictions above 0.5 to be positive while the rest are considered negative.

# Documentation

[tensorflow](https://www.tensorflow.org/guide), keras runs on top of tensorflow

[Keras](https://keras.io/) library used for building the neural network

[Adam](https://keras.io/optimizers/) optimizer

[sklearn](https://scikit-learn.org/stable/) library

[nltk](https://www.nltk.org/) library 

[Porter Stemmer](http://www.nltk.org/howto/stem.html) used for finding the stem of each word

[re](https://docs.python.org/3/library/re.html) for text cleaning (regex) 

[pandas](https://pandas.pydata.org/) library

# Library Installation

* pip install tensorflow
* pip install Keras
* pip install -U scikit-learn
* pip install nltk
* pip install regex
* pip install pandas

# Dataset and Inference

We got our dataset from [SuperDataScience](https://www.superdatascience.com/pages/machine-learning). The dataset used for this project can be found from the 'natural language processing' section of the link. 

The average accuracy yeilded by the model is **77%** and this can be drastically increased with a larger dataset to train our model. A screenshot of the confusion matrix is available in the repository(_Correct predictions are shown in the top left and bottom right cell of the matrix while the other two cells show the incorrect predictions_). 
