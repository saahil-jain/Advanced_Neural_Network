import Perceptron
import NNlayers
import Network

inputarr = [[2, 5] , [6, 8] , [9, 10] , [3, 7] , [2, 1]]
# mid = [[19, 10] , [36, 17] , [48, 22] , [27, 13] , [7, 6]]
outputarr = [[-1, 0] , [0, 1] , [1, 1] , [0, 1] , [-1, -1]]
# output[0] = 2*input[0] + 3*input[1]
# output[1] = 4*input[0] + 5*input[1] + 6

print("\n\nStart")
mynetwork = Network.Network(2, 0.005)
print("Input layer created")
mynetwork.addlayer(2, "relu")
print("Hidden layer created")
mynetwork.addlayer(2, "tanh")
print("All layers created\n\n")
for iterator in range(10000):
    index = iterator % 5
    mynetwork.train(inputarr[index],outputarr[index])

print(mynetwork.predict([2,5]))   # expected output = -1, 0
print(mynetwork.predict([6,8]))   # expected output = 0, 1
print(mynetwork.predict([9,10]))  # expected output = 1, 1
print(mynetwork.predict([3,7]))   # expected output = 0, 1
print(mynetwork.predict([2,1]))   # expected output = -1, -1
print(mynetwork.predict([5,100])) # expected output = 1, 1
print("\n\n\n")