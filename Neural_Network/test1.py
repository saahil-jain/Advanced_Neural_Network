import Perceptron
import NNlayers
import Network

inputarr = [[2, 5] , [6, 8] , [9, 10] , [3, 7] , [2, 1]]

outputarr = [[19] , [36] , [48] , [27] , [7]]  
# output[0] = 2*input[0] + 3*input[1]


print("\n\nStart")
mynetwork = Network.Network(2, 0.01)
print("Input layer created")
mynetwork.addlayer(1, "relu")
print("All layers created\n\n")
for iterator in range(2000):
    index = iterator % 5
    mynetwork.train(inputarr[index],outputarr[index])

print(mynetwork.predict([2,5]))   # expected output = 19
print(mynetwork.predict([6,8]))   # expected output = 36
print(mynetwork.predict([9,10]))  # expected output = 48
print(mynetwork.predict([3,7]))   # expected output = 27
print(mynetwork.predict([2,1]))   # expected output = 7
print(mynetwork.predict([5,100])) # expected output = 310
print("\n\n\n")