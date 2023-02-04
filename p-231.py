import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtext('books.csv', delimiter=',')
x = dataset.iloc[:,[4,11]].values
y = dataset.iloc[:,3].values
print("value of X are:", x)
print("value of Y are:", y)

model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()