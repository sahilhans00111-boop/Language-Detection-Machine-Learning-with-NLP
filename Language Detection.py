import numpy as np
import pandas as pd

data = pd.read_csv("language.csv")
data

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data
data.isnull().sum()
data['language'].value_counts()
data
data.dtypes
x = np.array(data['Text'])
y = np.array(data['language'])
print(x)
print(y)
cv = CountVectorizer()
x = cv.fit_transform(x)
x_train,x_test, y_train,y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)
x_train
print(x_train)
y_test
model = MultinomialNB()
model.fit(x_train,y_train)
model.score(x_test,y_test)

user = input("Enter a text")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)