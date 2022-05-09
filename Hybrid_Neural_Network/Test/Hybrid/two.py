import numpy as np
import random


tests = 100
iterations = 100
eta = 0.000000000005
inputarr = []
for _ in range(tests):
    x1 = random.randint(0,100)
    x2 = random.randint(0,100)
    inputarr.append([x1,x2])
outputarr = []
l = len(inputarr)
for t in inputarr:
    y = t[0]*t[0]#*t[0] + t[1]*t[1]
    outputarr.append(y)

w1 = random.uniform(-0.5,0.5)
w2 = random.uniform(-0.5,0.5)
w3 = random.uniform(-0.5,0.5)
w4 = random.uniform(-0.5,0.5)
w5 = random.uniform(-0.5,0.5)
w6 = random.uniform(-0.5,0.5)
w7 = random.uniform(-0.5,0.5)
b1 = random.uniform(-0.5,0.5)
b2 = random.uniform(-0.5,0.5)
b3 = random.uniform(-0.5,0.5)
b4 = random.uniform(-0.5,0.5)
for _ in range(iterations):
    index = 0
    for x in inputarr:
        x1,x2 = x[:]
        Td = outputarr[index]
        index+=1
        h1 = w1*x1 + w2*x2 + b1
        h2 = w3*x1 + w4*x2 + b2
        H = h1*h2 + b3
        y = w5*h1 +w6*h2 +w7*H + b4

        dy = -Td + y
        db4 = dy
        dw7 = H*dy
        dw6 = h2*dy
        dw5 = h1*dy

        dH = w7*dy
        db3 = dH
        dh2 = w6*dy + h1*dH
        db2 = dh2
        dh1 = w5*dy + h2*dH
        db1 = dh1

        dw1 = x1*dh1
        dw2 = x2*dh1
        dw3 = x1*dh2
        dw4 = x2*dh2

        w1 -= eta*dw1
        w2 -= eta*dw2
        w3 -= eta*dw3
        w4 -= eta*dw4
        w5 -= eta*dw5
        w6 -= eta*dw6
        w7 -= eta*dw7
        b1 -= eta*db1
        b2 -= eta*db2
        b3 -= eta*db3
        b4 -= eta*db4
index = 0
mae = 0
mse = 0
for x in inputarr:
        x1,x2 = x[:]
        Td = outputarr[index]
        index+=1
        h1 = w1*x1 + w2*x2 + b1
        h2 = w3*x1 + w4*x2 + b2
        H = h1*h2 + b3
        y = w5*h1 +w6*h2 +w7*H + b4
        print("{0:2.4f}".format(x[0]), ",", "{0:2.4f}".format(x[1]), "  | |", "{0:10.3f}".format(Td), "  | |", "{0:10.3f}".format(y))
        err = Td - y
        mae +=  abs(err)
        mse += err*err


print("{0:2.4f}".format(w1), "{0:2.4f}".format(w2), "{0:2.4f}".format(b1), "{0:2.4f}".format(b2), "{0:2.4f}".format(b3), sep="\t")
print("{0:2.4f}".format(w3), "{0:2.4f}".format(w4), "{0:2.4f}".format(w5), "{0:2.4f}".format(b4), sep="\t")
print("mae : {0:4.4f}".format(mae/tests))
print("mse : {0:4.4f}".format(mse/tests))