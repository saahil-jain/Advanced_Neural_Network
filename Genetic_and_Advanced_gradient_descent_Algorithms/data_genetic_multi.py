import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import random
import copy
random.seed(1)
SIZE = 8

# import cleaner as c
# import os
# os.remove("cleaned_data.csv")
# df = pd.read_csv ('data.csv')
# label = "reslt"
# clean = c.cleaner(df, label, ['a', 'age', 'weight1', 'HB', 'IFA', 'BP1', 'res'], ['history'])
# print(clean.result)
# clean.result.round(3)
# clean.result.to_csv("cleaned_data.csv", index = None, header=True)

startTime = datetime.now()
def normalize(x):
    result = x.copy()
    for feature_name in x.columns:
        max_value = x[feature_name].max()
        min_value = x[feature_name].min()
        result[feature_name] = (x[feature_name] - min_value) / (max_value - min_value)
    return result

features = ["a","age","weight1","history","HB","IFA","BP1","res"]

# reading in the csv as a dataframe
df = pd.read_csv('cleaned_data.csv')

df = normalize(df)
# selecting the features and target
X = df[features]
y = df['reslt']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)#, random_state = 42)

# Data Normalization
from sklearn.preprocessing import StandardScaler as SC
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import numpy as np
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# importing the required layers from keras
import Network
learningRate = 0.6
initialLearningRate = learningRate
mynetworks = []
# layering up the cnn
for i in range(SIZE):
    print("\n\nStart")
    mynetwork = Network.Network(8, learningRate)
    print("Input layer created")
    # mynetwork.addlayer(6, "relu")
    # print("Hidden layer created")
    mynetwork.addlayer(3, "relu")
    print("Hidden layer created")
    mynetwork.addlayer(1, "sigmoid")
    print("All layers created\n\n")
    mynetworks.append(mynetwork)

performance = []
error_rate = [1]*SIZE
error_rate_train = [1]*SIZE
error_rate_test = [1]*SIZE
iterator = 0

def train_test(mynetwork, threshold = 0.5):
    count_train = 0
    for i in range(len(X_train)):
        output = mynetwork.predict(list(X_train[i]))[0]
        o = output
        if o<threshold:
            o=0
        else:
            o=1
        if o != y_train[i]:
            # print(o," - ",y_train[i])
            count_train+=1
    error_rate_train = count_train/len(X_train)
    return error_rate_train

def testing(mynetwork, threshold = 0.5):
    count_train = 0
    count_test = 0
    total = len(X_train)+len(X_test)
    for i in range(len(X_train)):
        output = mynetwork.predict(list(X_train[i]))[0]
        o = output
        if o<threshold:
            o=0
        else:
            o=1
        if o != y_train[i]:
            count_train+=1
    for i in range(len(X_test)):
        output = mynetwork.predict(list(X_test[i]))[0]
        o = output
        if o<threshold:
            o=0
        else:
            o=1
        if o != y_test[i]:
            count_test+=1
    error_rate_test = count_test/len(X_test)
    error_rate_train = count_train/len(X_train)
    error_rate = (count_train+count_test)/total
    return [error_rate, error_rate_train, error_rate_test]


def jumplocalminima(mynetwork):
    print("\n\n TRYING TO JUMP OUT OF LOCAL MINIMA")
    mynetwork.globalreset()
    error_rate = train_test(mynetwork)
    # learningRate = initialLearningRate
    # mynetwork.changelearningrate(learningRate)
    return error_rate

def crossover(net1, net2):
    newnetwork = copy.deepcopy(net1)
    for _ in range(10):
        depth, node, weight, value = net2.get_random_weight()
        newnetwork.set_random_weight(depth, node, weight, value)
    return newnetwork

def generate_new_network(mynetworks):
    a = error_rate.index(min(error_rate))
    b = random.randint(0,SIZE-1)
    newnetwork = crossover(mynetworks[b],mynetworks[a])
    return newnetwork

# training
for i in range(SIZE):
    error_rate[i], error_rate_train[i], error_rate_test[i] = testing(mynetworks[i])
    print("\nstarting error :", "{:5.4f}".format(error_rate[i]), "\ttrain error :", "{:5.4f}".format(error_rate_train[i]), "\ttest error :", "{:5.4f}".format(error_rate_test[i]))

error_rate = error_rate_train
performance.append([iterator, min(error_rate)])
mutationNumber = 10
repetition = 0
while min(error_rate) >= 0.10:
    if repetition > 100:
        for i in range(SIZE):
            jumplocalminima(mynetworks[i])
            error_rate_new, error_rate_train_new, error_rate_test_new = testing(mynetworks[i])
            error_rate[i] = error_rate_train_new
        repetition = 0 

    iterator += 1
    newnetwork = generate_new_network(mynetworks)
    newnetwork.genomicweightchange(mutationNumber)
    error_rate1 = train_test(newnetwork)
    if error_rate1 <= max(error_rate):
        # repetition = 0
        index = error_rate.index(max(error_rate))
        error_rate[index] = error_rate1
        mynetworks[index] = newnetwork
    if error_rate1 < max(error_rate):
        repetition = 0
    else:
        repetition += 1
    performance.append([iterator, min(error_rate)])
    print("train error :", "{:5.4f}".format(min(error_rate)))

# using the learned weights to predict the target 

for i in range(SIZE):
    error_rate[i], error_rate_train[i], error_rate_test[i] = testing(mynetworks[i])
    print("\nfinal error :", "{:5.4f}".format(error_rate[i]), "\ttrain error :", "{:5.4f}".format(error_rate_train[i]), "\ttest error :", "{:5.4f}".format(error_rate_test[i]))

error_rate = error_rate_train
iterator += 1
performance.append([iterator, min(error_rate)])

print("\n*******************************************************************************************************************************************\n\nchanging threshold\n")

print("\n\nfinal accuracy :", "{:5.4f}".format((1-min(error_rate))*100), "\ttrain accuracy :", "{:5.4f}".format((1-min(error_rate_train))*100), "\ttest accuracy :", "{:5.4f}".format((1-min(error_rate_test))*100), "\n\n")

index = error_rate.index(min(error_rate))
# plotting a confusion matrix
confusionMatrix = [[0,0],[0,0]]
for i in range(len(X_train)):
    output = mynetworks[index].predict(list(X_train[i]))[0]
    o = output
    if o<0.5:
        o=0
    else:
        o=1
    if o == y_train[i] and o == 1:
        confusionMatrix[0][0] += 1
    elif o == y_train[i] and o == 0:
        confusionMatrix[1][1] += 1
    elif o != y_train[i] and o == 1:
        confusionMatrix[0][1] += 1
    else:
        confusionMatrix[1][0] += 1
for i in range(len(X_test)):
    output = mynetworks[index].predict(list(X_test[i]))[0]
    o = output
    if o<0.5:
        o=0
    else:
        o=1
    if o == y_test[i] and o == 1:
        confusionMatrix[0][0] += 1
    elif o == y_test[i] and o == 0:
        confusionMatrix[1][1] += 1
    elif o != y_test[i] and o == 1:
        confusionMatrix[0][1] += 1
    else:
        confusionMatrix[1][0] += 1

print("Confusion Matrix : ", confusionMatrix)

# printing execution time of script
print("\n")
print("Execution time in seconds = ", datetime.now() - startTime)

# import pickle
# with open("model", "wb") as file:
#     pickle.dump(mynetwork, file)

# plotting graph
x_val = [x[0] for x in performance[:]]
y_val = [x[1] for x in performance[:]]
plt.plot(x_val,y_val)
plt.plot(x_val,y_val,'or')
# plt.show()
plt.savefig('graphs/data_genetic_multi.png')
