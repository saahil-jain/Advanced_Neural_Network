from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random


tests = 100
iterations = 100
inputarr = [[]]
for _ in range(tests):
    x1 = random.randint(0,100)
    # x2 = random.randint(0,10)
    inputarr[0].append([x1])
outputarr = []
l = len(inputarr[0])
i = 0
for t in inputarr[0]:
    t1 = t[0]
    # t2 = t[1]
    y = t1*7 + 20
    outputarr.append([y])
    i+=1
print(inputarr[0])
print(outputarr[0])
outputarr = np.array(outputarr)


model = Sequential()
model.add(Dense(8, input_dim=1))
# model.add(Dense(3))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])


y_pred = model.predict(inputarr)
for i in range(len(y_pred)):
    print("{0:2.0f}".format(inputarr[0][i][0]), "  | |", "{0:10.3f}".format(outputarr[i][0]), "  | |", "{0:10.3f}".format(y_pred[i][0]))
model.fit(inputarr, outputarr, epochs=iterations, batch_size=1)

# y_pred = model.predict(inputarr)
# print("TRAIN DATA\n")
# for i in range(len(y_pred)):
#     print("{0:2.0f}".format(inputarr[0][i][0]), "  | |", "{0:10.3f}".format(outputarr[i][0]), "  | |", "{0:10.3f}".format(y_pred[i][0]))
# print("\n\n\n")


# TEST

# test_size = 10
# test_data = [[]]
# i = 0
# while i<test_size:
#     x1 = random.randint(0,100)
#     # x2 = random.randint(0,10)
#     elem = [x1]
#     if elem not in inputarr[0]:
#         test_data[0].append(elem)
#         i += 1
# out = []
# l = len(test_data[0])
# i = 0
# for t in test_data[0]:
#     t1 = t[0]
#     # t2 = t[1]
#     y = t1*7 + 20
#     out.append([y])
#     i+=1
# out = np.array(out)
# y_pred = model.predict(test_data)

# print("TEST DATA\n")
# for i in range(len(test_data[0])):
#     print("{0:2.0f}".format(test_data[0][i][0]), "  | |", "{0:10.3f}".format(out[i][0]), "  | |", "{0:10.3f}".format(y_pred[i][0]))
# print("\n\n\n")