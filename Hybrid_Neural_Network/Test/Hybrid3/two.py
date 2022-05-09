import numpy as np
import random


tests = 100
iterations = 100
eta = 0.0000000000005
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
w8 = random.uniform(-0.5,0.5)
w9 = random.uniform(-0.5,0.5)
w10 = random.uniform(-0.5,0.5)
b1 = random.uniform(-0.5,0.5)
b2 = random.uniform(-0.5,0.5)
b3 = random.uniform(-0.5,0.5)
b4 = random.uniform(-0.5,0.5)
b5 = random.uniform(-0.5,0.5)
for _ in range(iterations):
    index = 0
    for x in inputarr:
        x1,x2 = x[:]
        Td = outputarr[index]
        index+=1
        h1 = w1*x1 + w2*x2 + b1
        h2 = w3*x1 + w4*x2 + b2
        h3 = w5*x1 + w6*x2 + b3
        H = h1*h2*h3 + b4
        y = w7*h1 + w8*h2 + w9*h2 + w10*H + b5

        dy = -Td + y
        db5 = dy
        dw10 = H*dy
        dw9 = h3*dy
        dw8 = h2*dy
        dw7 = h1*dy

        dH = w10*dy
        db4 = dH
        dh3 = w9*dy + h1*h2*dH
        db3 = dh3
        dh2 = w8*dy + h1*h3*dH
        db2 = dh2
        dh1 = w7*dy + h2*h3*dH
        db1 = dh1

        dw6 = x2*dh3
        dw5 = x1*dh3
        dw4 = x2*dh2
        dw3 = x1*dh2
        dw2 = x2*dh1
        dw1 = x1*dh1

        w1 -= eta*dw1
        w2 -= eta*dw2
        w3 -= eta*dw3
        w4 -= eta*dw4
        w5 -= eta*dw5
        w6 -= eta*dw6
        w7 -= eta*dw7
        w8 -= eta*dw8
        w9 -= eta*dw9
        w10 -= eta*dw10
        b1 -= eta*db1
        b2 -= eta*db2
        b3 -= eta*db3
        b4 -= eta*db4
        b5 -= eta*db5
index = 0
mae = 0
mse = 0
for x in inputarr:
        # h1 = w1*x1 + w2*x2 + b1
        # h2 = w3*x1 + w4*x2 + b2
        # H = h1*h2 + b3
        # y = w5*h1 +w6*h2 +w7*H + b4
        x1,x2 = x[:]
        Td = outputarr[index]
        index+=1
        h1 = w1*x1 + w2*x2 + b1
        h2 = w3*x1 + w4*x2 + b2
        h3 = w5*x1 + w6*x2 + b3
        H = h1*h2*h3 + b4
        y = w7*h1 + w8*h2 + w9*h2 + w10*H + b5
        print("{0:2.4f}".format(x[0]), ",", "{0:2.4f}".format(x[1]), "  | |", "{0:10.3f}".format(Td), "  | |", "{0:10.3f}".format(y))
        err = Td - y
        mae +=  abs(err)
        mse += err*err


# print("{0:2.4f}".format(w1), "{0:2.4f}".format(w2), "{0:2.4f}".format(b1), "{0:2.4f}".format(b2), "{0:2.4f}".format(b3), sep="\t")
# print("{0:2.4f}".format(w3), "{0:2.4f}".format(w4), "{0:2.4f}".format(w5), "{0:2.4f}".format(b4), sep="\t")
print("mae : {0:4.4f}".format(mae/tests))
print("mse : {0:4.4f}".format(mse/tests))