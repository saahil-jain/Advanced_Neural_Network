import Perceptron
import NNlayers
import Network

inputarr = [[2, 5] , [6, 8] , [9, 10] , [3, 7] , [2, 1]]
# mid = [19, 36, 48, 27, 7]
outputarr = [[19, 38] , [36, 72] , [48, 96] , [27, 54] , [7, 14]]
# output[0] = 2*input[0] + 3*input[1]
# output[1] = 4*input[0] + 6*input[1]

print("\n\nStart")
mynetwork = Network.Network(2, 0.00005)
print("Input layer created")
mynetwork.addlayer(1, "relu")
print("Hidden layer created")
mynetwork.addlayer(2, "relu")
print("All layers created\n\n")
for iterator in range(5000):
    index = iterator % 5
    mynetwork.train(inputarr[index],outputarr[index])

print(mynetwork.predict([2,5]))   # expected output = 19, 38
print(mynetwork.predict([6,8]))   # expected output = 36, 72
print(mynetwork.predict([9,10]))  # expected output = 48, 96
print(mynetwork.predict([3,7]))   # expected output = 27, 54
print(mynetwork.predict([2,1]))   # expected output = 7, 14
print(mynetwork.predict([5,100])) # expected output = 310, 620
print("\n\n\n")