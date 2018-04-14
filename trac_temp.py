'''
install the following dependencies
• numpy => pip install numpy
• sklearn => pip install scikit-learn
• nltk => pip install nltk
• pandas => pip install pandas
'''

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.classify.scikitlearn import SklearnClassifier
import nltk
# from matplotlib import pyplot as plt

# this takes train data from the file agr_en_dev.csv
df_train = pd.read_csv('agr_en_train.csv', sep=',', names=['id', 'comment', 'category'])

df_train.loc[df_train["category"] == 'NAG', "category", ] = 0
df_train.loc[df_train["category"] == 'CAG', "category", ] = 1
df_train.loc[df_train["category"] == 'OAG', "category", ] = 2

df_train_x = df_train["comment"]
df_train_y = df_train["category"]

# google train_test_split() to understand the functionality of train_test_split()
x_train, x_test_ignore, y_train, y_test_ignore = train_test_split(df_train_x, df_train_y, test_size=0.0, random_state=4)
# THIS PART MUST BE TESTED ONCE
# df_train = pd.read_csv('agr_en_train.csv', sep=',', names=['id', 'comment', 'category'])

# df_train.loc[df_train["category"] == 'NAG', "category", ] = 0
# df_train.loc[df_train["category"] == 'CAG', "category", ] = 1
# df_train.loc[df_train["category"] == 'OAG', "category", ] = 2

# df_train_x = df_train["comment"]
# df_train_y = df_train["category"]

# # google train_test_split() to understand the functionality of train_test_split()
# x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.0, random_state=4)

# this takes test data from the file agr_en_dev.csv
df_test = pd.read_csv('agr_en_dev.csv', sep=',', names=['id', 'comment', 'category'])

df_test.loc[df_test["category"] == 'NAG', "category", ] = 0
df_test.loc[df_test["category"] == 'CAG', "category", ] = 1
df_test.loc[df_test["category"] == 'OAG', "category", ] = 2

df_test_x = df_test["comment"]
df_test_y = df_test["category"]

x_train_ignore, x_test, y_train_ignore, y_test = train_test_split(df_test_x, df_test_y, test_size=0.99999, random_state=4)

# TfidfVectorizer is used to determine the frequency of the word in the document
cv1 = TfidfVectorizer(min_df=1, stop_words='english')

# tfidf of training data is created
x_traincv = cv1.fit_transform(x_train)

a = x_traincv.toarray()
# print(len(x_test))

# tfidf of training data is created
x_testcv = cv1.transform(x_test)
x_testcv.toarray()

# object of Multinomial Naive Bayes classifier is created
mnb = MultinomialNB()

y_train = y_train.astype('int')

# model is tained
mnb.fit(x_traincv, y_train)

# predictions are done
predictions_mnb = mnb.predict(x_testcv)

a = np.array(y_test)

count_mnb = 0

#b = np.array(df_test_x)
for i in range(len(predictions_mnb)):
    if predictions_mnb[i] == a[i]:
        count_mnb = count_mnb + 1

total = len(predictions_mnb)

# accuracy is calculated
accuracy = count_mnb / float(total)

print('\n\tMultinomial Naive Bayes Accuracy = ', accuracy)

lr = LogisticRegression()
lr.fit(x_traincv, y_train)

predictions_lr = lr.predict(x_testcv)

total_lr = len(predictions_lr)

a = np.array(y_test)

count_lr = 0

for i in range(len(predictions_lr)):
    if predictions_lr[i] == a[i]:
        count_lr = count_lr + 1

# accuracy is calculated
accuracy_lr = count_lr / total_lr

print('\tLogistic Regression accuracy =', accuracy_lr, '\n')
