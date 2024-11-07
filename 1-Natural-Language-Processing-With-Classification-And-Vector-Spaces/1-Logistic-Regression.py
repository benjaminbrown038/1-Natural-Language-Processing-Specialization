import nltk 
from os import getcwd
import w1_unittest

import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
from utils import process_tweet, build_freqs


nltk.download('twitter_samples')
nltk.download('stopwords')

filePath = f""



all_positive_tweets
all_negative_tweets

test_pos
train_pos
test_neg
train_neg

train_x
test_x

train_y
test_y

print
print

freqs
print
print

print
print

def sigmoid():

if:
else:
if:
else:

w1_unittest.test_sigmoid()

def gradientDescent():

    return J, theta


np.random.seed()
tmp_X
tmp_Y
tmp_J, tmp_theta = gradientDescent()
print
print

w1_unittest.test_gradientDescent(gradientDescent)

def extract_features(tweet,freqs,process_tweet=process_tweet):
    return x 

tmp1 = extract_features()
print()

tmp2 = extract_features()
print

w1_unittest.test_extract_features()

X = np.zeros()
for i in range(len(train_x)):
    X[i] = 

Y = train_y
J, theta
print
print

def predict_tweet():
    return y_pred

for tweet in:
  print

my_tweet = 
predict_tweet

w1_unittest.test_predict_tweet(predict_tweet,freqs,theta)

def test_logistic_regression(test_x,test_y,freqs,theta,predict_tweet = predict_tweet):
    return accuracy

tmp_accuracy = test_logistic_regression()
print


