import Perceptron
import NNlayers
import Network
import matplotlib.pyplot as plt

inputarr = [[2, 5] , [6, 8] , [9, 10] , [3, 7] , [2, 1]]

outputarr = [[0] , [1] , [1] , [1] , [0]]
# output[0] = 2*input[0] + 3*input[1] > 25 ? 1:0


print("\n\nStart")
mynetwork = Network.Network(2, 1.4, "ADA", True)
print("Input layer created")
mynetwork.addlayer(1, "sigmoid")
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
plt.savefig('graphs/test3ada.png')

print(mynetwork.predict([2,5]))   # expected output = 0
print(mynetwork.predict([6,8]))   # expected output = 1
print(mynetwork.predict([9,10]))  # expected output = 1
print(mynetwork.predict([3,7]))   # expected output = 1
print(mynetwork.predict([2,1]))   # expected output = 0
print(mynetwork.predict([5,100])) # expected output = 1
print("\n\n\n")
print(total_error)
print("\n\n\n")