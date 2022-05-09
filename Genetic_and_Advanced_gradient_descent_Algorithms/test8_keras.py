from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random


tests = 100
inputarr = [[]]
for _ in range(tests):
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1)
    inputarr[0].append([x1,x2])
outputarr = []
for t in inputarr[0]:
    y = t[0] + 20*t[1]
    outputarr.append([y])
outputarr = np.array(outputarr)


model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Dense(4))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])


y_pred = model.predict(inputarr)
for i in range(len(y_pred)):
    print("{0:2.0f}".format(inputarr[0][i][0]), ",", "{0:2.0f}".format(inputarr[0][i][1]), "  | |", "{0:10.3f}".format(outputarr[i][0]), "  | |", "{0:10.3f}".format(y_pred[i][0]))
model.fit(inputarr, outputarr, epochs=50, batch_size=1)

y_pred = model.predict(inputarr)
print("TRAIN DATA\n")
train_error = 0
for i in range(len(y_pred)):
    error = outputarr[i][0] - y_pred[i][0]
    train_error += error * error
    print("{0:2.0f}".format(inputarr[0][i][0]), ",", "{0:2.0f}".format(inputarr[0][i][1]), "  | |", "{0:10.3f}".format(outputarr[i][0]), "  | |", "{0:10.3f}".format(y_pred[i][0]))
print("\n\n\n")

test_size = 100
test_data = [[]]
i = 0
while i<test_size:
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1)
    elem = [x1,x2]
    if elem not in inputarr[0]:
        test_data[0].append(elem)
        i += 1
out = []
i = 0
for t in test_data[0]:
    y = t[0] + 20*t[1]
    out.append([y])
    i+=1
out = np.array(out)
y_pred = model.predict(test_data)

print("TEST DATA\n")
test_error = 0
for i in range(len(test_data[0])):
    error = out[i][0] - y_pred[i][0]
    test_error += error * error
    print("{0:2.0f}".format(test_data[0][i][0]), ",", "{0:2.0f}".format(test_data[0][i][1]), "  | |", "{0:10.3f}".format(out[i][0]), "  | |", "{0:10.3f}".format(y_pred[i][0]))
print("\n\n\n")

print("Train Error :", train_error/tests)
print("Test Error  :", test_error/test_size)
print("\n\n\n")