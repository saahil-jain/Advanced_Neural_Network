import Perceptron
import NNlayers
import Network
import matplotlib.pyplot as plt

inputarr = [[2, 5] , [6, 8] , [9, 10] , [3, 7] , [2, 1]]
# mid = [19, 36, 48, 27, 7]
outputarr = [[19, 38] , [36, 72] , [48, 96] , [27, 54] , [7, 14]]
# output[0] = 2*input[0] + 3*input[1]
# output[1] = 4*input[0] + 6*input[1]

print("\n\nStart")
mynetwork = Network.Network(2, 0.5, "ADA")
print("Input layer created")
mynetwork.addlayer(1, "relu")
print("Hidden layer created")
mynetwork.addlayer(2, "relu")
print("All layers created\n\n")
performance = []
for iterator in range(40):
    total_error = 0
    for index in range(5):
        mynetwork.train(inputarr[index],outputarr[index])
        output = mynetwork.predict(inputarr[index])
        error = abs(output-outputarr[index])
        squared_error = 0
        for e in error:
            squared_e = e*e
            squared_error += squared_e
        total_error += squared_error
    performance.append([iterator, total_error])
x_val = [x[0] for x in performance[:]]
y_val = [x[1] for x in performance[:]]
plt.plot(x_val,y_val)
plt.plot(x_val,y_val,'or')
# plt.show()
plt.savefig('graphs/test2ada.png')

print(mynetwork.predict([2,5]))   # expected output = 19, 38
print(mynetwork.predict([6,8]))   # expected output = 36, 72
print(mynetwork.predict([9,10]))  # expected output = 48, 96
print(mynetwork.predict([3,7]))   # expected output = 27, 54
print(mynetwork.predict([2,1]))   # expected output = 7, 14
print(mynetwork.predict([5,100])) # expected output = 310, 620
print("\n\n\n")
print(total_error)
print("\n\n\n")