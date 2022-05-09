import Perceptron
import NNlayers
import Network

inputarr = [[2, 5] , [6, 8] , [9, 10] , [3, 7] , [2, 1]]

outputarr = [[0] , [1] , [1] , [1] , [0]]
# output[0] = 2*input[0] + 3*input[1] > 25 ? 1:0


print("\n\nStart")
mynetwork = Network.Network(2, 0.1)
print("Input layer created")
mynetwork.addlayer(1, "sigmoid")
print("All layers created\n\n")
for iterator in range(5000):
    index = iterator % 5
    mynetwork.train(inputarr[index],outputarr[index])

print(mynetwork.predict([2,5]))   # expected output = 0
print(mynetwork.predict([6,8]))   # expected output = 1
print(mynetwork.predict([9,10]))  # expected output = 1
print(mynetwork.predict([3,7]))   # expected output = 1
print(mynetwork.predict([2,1]))   # expected output = 0
print(mynetwork.predict([5,100])) # expected output = 1
print("\n\n\n")