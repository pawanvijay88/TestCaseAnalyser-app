# Importing required libraries
import pandas as pd
import numpy as np
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Reading a dataset
df = pd.read_csv('Bluetooth_TestCases.csv')
pd.set_option('display.max_columns', df.columns.shape[0])
df = df.drop(['ID', 'URL', 'Priority', 'Name',
       'Domain', 'Created By', 'Variant1', 'Test Type', 'Variant2',
       'DO NOT UPDATE - Legacy Requirements Set Name 1',
       'DO NOT UPDATE - Legacy Requirements Set Name 2'], axis=1)
df.rename(columns={'Automation State': 'Automation_State'}, inplace=True)
df_test = df[df['Automation_State'] == 'Pending']
df_test.to_csv('test.csv')
df.drop(df[df['Automation_State'] == 'Pending'].index, inplace=True)
df.reset_index(inplace=True, drop=True)
df.replace("Can't be Automated", "Non_Automatable", inplace=True)

# Text Preprocessing
corpus = []
wnl = WordNetLemmatizer()
for i in range(0, df.shape[0]):
       review = re.sub('[^a-zA-Z]', ' ', str(df['Description'][i]))
       review = review.lower()
       review = review.split()
       review = [wnl.lemmatize(word)for word in review if word not in set(stopwords.words('english'))]
       review = ' '.join(review)
       corpus.append(review)

tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
X = tfidf.fit_transform(corpus).toarray()
print(type(X))
df['Automation_State'] = pd.get_dummies(df['Automation_State'], drop_first=True) # 1: Non_Automatable, 0: Automatable
y = df.iloc[:, 1].values

# Creating a pickle file for the TfidfVectorizer
pickle.dump(tfidf, open('tfidf-transform.pkl', 'wb'))

# Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Accuracy Score = {}'.format(accuracy_score(y_test, y_pred)))

# Creating a pickle file for the Multinomial Naive Bayes model
pickle.dump(classifier, open('model.pkl', 'wb'))