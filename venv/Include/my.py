import numpy as np
import matplotlib.pyplot as plt
import math

def getGuess(number):
    mu = 0
    sigma = 0.1
    np.random.seed(15)
    return np.random.normal(mu, sigma, number)

def getLSM(x_coordinate,y_coordinate,numberOfPoly):
    #LSM method should be  matX*matA = matY
    #get matX
    matX = []
    for i in range(0,numberOfPoly+1):
        mat = []
        for j in range(i,numberOfPoly+i+1):
            sum = 0
            for x in x_coordinate:
                sum = sum + x**j
            mat.append(sum)
        matX.append(mat)
    # matX = numpy.array(matX)
    #get matY
    matY = []
    for i in range(0,numberOfPoly+1):
        sum = 0
        for x,y in zip(x_coordinate,y_coordinate):
            sum = sum + x**i*y
        matY.append(sum)
    matA = []
    matA = np.linalg.solve(matX,matY)
    print("length of it \n")
    print(len(matA))
    return matA


#get the numebr of trainning data
number = input("please input the number of data!\n")
number = int(number)
numberOfPoly = input("please input the poly number\n")
numberOfPoly= int(numberOfPoly)
singlePace = 2*math.pi/number
x_coordinate = np.linspace(0,2*math.pi,number)
y_coordinate = []
guessArray = getGuess(number)
for i in range(0,number):
    y_coordinate.append(math.sin(x_coordinate[i])+guessArray[i])
mat = getLSM(x_coordinate,y_coordinate,numberOfPoly)
y = 0
x = np.linspace(0, 2*math.pi, number)
for i in range(0,numberOfPoly+1):
    y = y + mat[i]*x**(numberOfPoly-i)
plt.scatter(x, y)
plt.plot(x, y, color='g')
plt.show()