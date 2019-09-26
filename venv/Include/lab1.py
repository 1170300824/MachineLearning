import numpy as np
import matplotlib.pyplot as plt
import math

def getGuess():
    mu = 0
    sigma = 0.1
    np.random.seed(15)
    return np.random.normal(mu, sigma, number)


def getMatX():
    # matX * matA = matY
    matX = []
    for i in range(0, numberOfPoly + 1):
        matLine = []
        for j in range(i, numberOfPoly + 1 + i):
            sum = 0
            for k in x_coordinate:
                sum += k ** j
            matLine.append(sum)
        matX.append(matLine)
    matX = np.array(matX)
    return matX

def getMatY():
    # matX * matA = matY
    matY = []
    for i in range(0, numberOfPoly + 1):
        sum = 0
        for m, n in zip(x_coordinate, y_coordinate):
            sum += m ** i * n
        matY.append(sum)
    return matY

def getLSM():
    matX = getMatX(x_coordinate,numberOfPoly)
    matY = getMatY(x_coordinate,y_coordinate,numberOfPoly)
    matA = np.linalg.solve(matX,matY)
    return matA

def getPunishLSM():
    # matX * matA = matY
    matX = []
    punishment = 1
    for i in range(0, numberOfPoly + 1):
        matLine = []
        for j in range(i, numberOfPoly + 1 + i):
            sum = 0
            for k in x_coordinate:
                sum += k ** j
                if(i == j):
                    sum = 2*punishment
            matLine.append(sum)
        matX.append(matLine)
    matX = np.array(matX)
    matY = []
    for i in range(0, numberOfPoly + 1):
        sum = 0
        for m, n in zip(x_coordinate, y_coordinate):
            sum += m ** i * n
        matY.append(sum)

    matA = np.linalg.solve(matX, matY)
    return matA

def check(matA):
    maxError = 0.05
    error = 0
    for x,y in zip(x_coordinate,y_coordinate):
        sum = y
        for i in range(0,numberOfPoly+1):
            sum -= matA[i]*(x**i)
        sum = sum**2
        error += sum


    error = error/len(x_coordinate)
    print(error)

    if error < 0.01:

        return False
    else :

        return True

def getGradient(matA):
    gradientArray = []
    for i in range(0,numberOfPoly+1):
        gradient = 0
        for x,y in zip(x_coordinate,y_coordinate):
            temp = 2*x**i
            sum = 0
            for j in range(0,numberOfPoly+1):
                sum += matA[j]*x**j
            sum -= y
            temp = temp*sum
            gradient += temp
        gradientArray.append(gradient/number)
    print(gradientArray)
    return gradientArray


def getGD():
    matA = np.array([])
    step = 0.0001
    for i in range(0,numberOfPoly+1):
        matA = np.concatenate((matA, [0]))
    while check(matA):
        matG = getGradient(matA)
        for i in range(0,numberOfPoly+1):
            matA[i] = matA[i]-step*matG[i]
    return matA

def check1(matA):
    maxError = 0.05
    error = 0
    for x,y in zip(x_coordinate,y_coordinate):
        sum = y
        for i in range(0,numberOfPoly+1):
            sum -= matA[i]*(x**i)
        sum = sum**2
        error += sum


    error = error/len(x_coordinate)
    return error

def getCGD():
    matX = np.array(getMatX())
    matY = np.array(getMatY())
    matA = np.array([])
    for i in range(0,numberOfPoly+1):
        matA = np.concatenate((matA, [0]))
    matY = np.transpose(matY)
    matA = np.transpose(matA)
    matr = matY - np.dot(matX,matA)
    matp = matr
    count = 0
    while (check(matA)):
        alpha = np.dot(matr, matr) / np.dot(np.dot(matX, matp), matp)
        matA = matA + alpha*matp
        tempMat = matr
        matr = matr - alpha*np.dot(matX,matp)
        matp = matr + np.dot(matr,matr)*matp/np.dot(tempMat,tempMat)
        count = count + 1
        if count > 500:
            break
    return matA

number = input("please input the number of data!\n")
number = int(number)
numberOfPoly = input("please input the poly number\n")
numberOfPoly= int(numberOfPoly)
x_coordinate = np.linspace(0,2*math.pi,number)
y_coordinate = []
guessArray = getGuess()
for i in range(0,number):
    y_coordinate.append(math.sin(x_coordinate[i])+guessArray[i])
mat = getCGD()
y = 0
x = np.linspace(0, 2*math.pi, 1000)
for i in range(0,numberOfPoly+1):
    y = y + mat[i]*x**(i)
plt.scatter(x_coordinate, y_coordinate)
plt.plot(x, y, color='g')
plt.show()