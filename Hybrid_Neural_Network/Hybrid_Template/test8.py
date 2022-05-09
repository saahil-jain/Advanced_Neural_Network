import Perceptron
import NNlayers
import Network
import matplotlib.pyplot as plt
import random
tests = 100
inputarr = []
for _ in range(tests):
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1)
    inputarr.append([x1,x2])
outputarr = []
for t in inputarr:
    y = t[0] + 20*t[1]
    outputarr.append([y])

print("\n\nStart")
mynetwork = Network.Network(2, 0.002, "GD")
print("Input layer created")
mynetwork.addlayer(2, "relu", True)
print("Hidden layer created")
mynetwork.addlayer(1, "relu")
print("All layers created\n\n")
performance = []
for iterator in range(50):
    total_error = 0
    for index in range(tests):
        mynetwork.train(inputarr[index],outputarr[index])
        output = mynetwork.predict(inputarr[index])
        error = abs(output-outputarr[index])
        squared_error = 0
        for e in error:
            squared_e = e*e
            squared_error += squared_e
        total_error += squared_error
    print("{0:4.0f}".format(iterator), " - ",total_error/tests)
    performance.append([iterator, total_error])
print(total_error)
x_val = [x[0] for x in performance[:]]
y_val = [x[1] for x in performance[:]]
plt.plot(x_val,y_val)
plt.plot(x_val,y_val,'or')
# plt.show()
plt.savefig('graphs/test8.png')

for i in range(tests):
    output = mynetwork.predict(inputarr[i])
    print("{0:2.0f}".format(inputarr[i][0]), ",", "{0:2.0f}".format(inputarr[i][1]), "  | |", "{0:10.3f}".format(outputarr[i][0]), "  | |", "{0:10.3f}".format(output[0]))
print("\n\n\n")
print("Train Error :", total_error/tests)

test_size = 100
test_data = []
i = 0
while i<test_size:
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1)
    elem = [x1,x2]
    if elem not in inputarr[0]:
        test_data.append(elem)
        i += 1
out = []
for t in test_data:
    y = t[0] + 20*t[1]
    out.append([y])
for index in range(test_size):
    output = mynetwork.predict(test_data[index])
    error = abs(output-out[index])
    squared_error = 0
    for e in error:
        squared_e = e*e
        squared_error += squared_e
    total_error += squared_error
print("Test Error  :",total_error/test_size)
print("\n\n\n")