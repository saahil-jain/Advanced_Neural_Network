import Perceptron
import NNlayers
import Network
import matplotlib.pyplot as plt

inputarr = [[2, 5] , [6, 8] , [9, 10] , [3, 7] , [2, 1]]
# mid = [[19, 10] , [36, 17] , [48, 22] , [27, 13] , [7, 6]]
outputarr = [[19, 39] , [36, 70] , [48, 92] , [27, 53] , [7, 16]]
# output[0] = 2*input[0] + 3*input[1]
# output[1] = 4*input[0] + 5*input[1] + 6

print("\n\nStart")
mynetwork = Network.Network(2, 0.5, "ADA")
print("Input layer created")
mynetwork.addlayer(2, "relu")
print("Hidden layer created")
mynetwork.addlayer(2, "relu")
print("All layers created\n\n")
performance = []
for iterator in range(20):
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
plt.savefig('graphs/test5ada.png')

print(mynetwork.predict([2,5]))   # expected output = 19, 39
print(mynetwork.predict([6,8]))   # expected output = 36, 70
print(mynetwork.predict([9,10]))  # expected output = 48, 92
print(mynetwork.predict([3,7]))   # expected output = 27, 53
print(mynetwork.predict([2,1]))   # expected output = 7, 16
print(mynetwork.predict([5,100])) # expected output = 310, 618
print("\n\n\n")
print(total_error)
print("\n\n\n")