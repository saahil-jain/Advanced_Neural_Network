import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import random
random.seed(1)

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
# layering up the cnn
print("\n\nStart")
mynetwork = Network.Network(8, learningRate)
print("Input layer created")
# mynetwork.addlayer(6, "relu")
# print("Hidden layer created")
mynetwork.addlayer(3, "relu")
print("Hidden layer created")
mynetwork.addlayer(1, "sigmoid")
print("All layers created\n\n")

performance = []
error_rate = 1
iterator = 0

def train_test(threshold = 0.5):
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

def testing(threshold = 0.5):
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


def jumplocalminima():
    print("\n\n TRYING TO JUMP OUT OF LOCAL MINIMA")
    mynetwork.globalreset()
    error_rate = train_test()
    # learningRate = initialLearningRate
    # mynetwork.changelearningrate(learningRate)
    return error_rate


# training
error_rate, error_rate_train, error_rate_test = testing()
print("starting error :", "{:5.4f}".format(error_rate), "\ttrain error :", "{:5.4f}".format(error_rate_train), "\ttest error :", "{:5.4f}".format(error_rate_test), "\n\n")

error_rate = error_rate_train
# mutationNumber = 34
repeatcounter = 0
saturationCounter = 0
while error_rate >= 0.1:
    if repeatcounter > 30 or  saturationCounter == 30:
        error_rate = jumplocalminima()
        saturationCounter = 0
        repeatcounter = 0

    nexthop = iterator+100
    mutationNumber = 1#mutationNumber - (10*mutationNumber)//100
    saturationCounter = 0
    while iterator<nexthop:
        mynetwork.genomicweightchange(mutationNumber)
        error_rate1 = train_test()
        if error_rate1 <= error_rate:
            if error_rate1 == error_rate:
                repeatcounter += 1
            else:
                repeatcounter = 0
            saturationCounter = 0
            error_rate = error_rate1
            mynetwork.confirmweightchange()
            performance.append([iterator, error_rate])
            iterator += 1
            print(iterator," - ", end = "") 
            print("error :", "{:5.4f}".format(error_rate), "\n\n")
        else:
            mynetwork.revertweightchange()
            saturationCounter += 1
        if saturationCounter == 30:
            break
        if repeatcounter > 30:
            break

# using the learned weights to predict the target 
error_rate, error_rate_train, error_rate_test = testing()
print("\nfinal error :", "{:5.4f}".format(error_rate), "\ttrain error :", "{:5.4f}".format(error_rate_train), "\ttest error :", "{:5.4f}".format(error_rate_test), "\n\n")

print("\n*******************************************************************************************************************************************\n\nchanging threshold\n")

# setting a confidence threshhold
final_threshold = 0
min_test_error = 1
min_train_error = 1
min_error = 1
for threshold in range(30,90):
    threshold /= 100
    error_rate, error_rate_train, error_rate_test = testing(threshold)
    print("threshold :", "{:3.2f}\t".format(threshold),"error :", "{:5.4f}".format(error_rate), "\ttrain error :", "{:5.4f}".format(error_rate_train), "\ttest error :", "{:5.4f}".format(error_rate_test), "\n\n")
    
    if error_rate_train<min_train_error:
        final_threshold = threshold
        min_test_error = error_rate_test
        min_train_error = error_rate_train
        min_error = error_rate
    elif error_rate_train == min_train_error:
        if error_rate_test<=min_test_error:
            final_threshold = threshold

error_rate, error_rate_train, error_rate_test = testing(final_threshold)
print("final threshold :", "{:3.2f}\t".format(final_threshold), "final error :", "{:5.4f}".format(error_rate), "\ttrain error :", "{:5.4f}".format(error_rate_train), "\ttest error :", "{:5.4f}".format(error_rate_test), "\n\n")
performance.append([iterator, error_rate_train])
iterator += 1
print("\n\nfinal accuracy :", "{:5.4f}".format((1-error_rate)*100), "\ttrain accuracy :", "{:5.4f}".format((1-error_rate_train)*100), "\ttest accuracy :", "{:5.4f}".format((1-error_rate_test)*100), "\n\n")


# plotting a confusion matrix
confusionMatrix = [[0,0],[0,0]]
for i in range(len(X_train)):
    output = mynetwork.predict(list(X_train[i]))[0]
    o = output
    if o<final_threshold:
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
    output = mynetwork.predict(list(X_test[i]))[0]
    o = output
    if o<final_threshold:
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
plt.savefig('graphs/data_genetic.png')
