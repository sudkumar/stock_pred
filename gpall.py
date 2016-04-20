import numpy as np
from sklearn import gaussian_process 
import pylab
import math

# Read the data file in memory
print("Reading data in memory....")
data = []
with open("data.csv") as fp:
    next(fp)
    for line in fp:
        x = list(map(float, line.strip().split(",")[1:5]))
        data.append(x)

print("Reading data done.")
data = data[:1500]
dataSize = len(data)
trainingDataSize = 1000
windowSize = 50

# create the training data from input
# X is a (trainingDataSize-windowSize)X(windowSize) matrix
# and Y is a column matrix of (trainingDataSize-windowSize) rows 

# initialize the matrixes with all zeros entries
print("Creating Training Dataset...")
X = [[] for j in range(trainingDataSize-windowSize)]
Y = [0 for i in range(trainingDataSize-windowSize)]

for i in range(trainingDataSize-windowSize):
    
    # update each item in X[i]
    currWindow = data[i:(i+windowSize)]
    for j in range(windowSize):
        X[i] += currWindow[j] 
    # get the close price for Y
    Y[i] = data[i+windowSize][3]

X_Train = np.array(X, dtype=np.float64)
Y_Train = np.array(Y, dtype=np.float64)

print("Creating Training Dataset done.")

# create the test data set
print("Creating Test Dataset...")
X = [[] for j in range(dataSize-trainingDataSize-windowSize)]
Y = [0 for i in range(dataSize-trainingDataSize-windowSize)]

for i in range(dataSize-trainingDataSize-windowSize):
    currWindow = data[(trainingDataSize+i):(trainingDataSize+i+windowSize)]
    for j in range(windowSize):
        X[i] += currWindow[j] 
    Y[i] = data[trainingDataSize+i+windowSize][3]

X_Test = np.array(X, dtype=np.float64)
Y_Test = np.array(Y, dtype=np.float64)
del X
del Y

print("Creating Test Dataset done.")

# Instantiate with corr as squared exponential
# corr= 'absolute_exponential' || 'squared_exponential' || 'generalized_exponential' || 'cubic' || 'linear'
for corr in ["absolute_exponential", "squared_exponential", "linear"]:
    print("Instantiating GP for "+corr)
    # nugget = 10* MACHINE_EPSILON
    # gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1,corr=corr)
    gp = gaussian_process.GaussianProcess(theta0=1e-1,corr=corr)
    # let gp learn

    print("Now let the Gaussian Process learn")
    gp.fit(X_Train, Y_Train)
    # print("Gaussian is ready to use")

    # Now do the prediction
    print("Gaussian is predecting")
    Y_Pred, Sigma2_Pred = gp.predict(X_Test, eval_MSE=True)
    ylength = len(Y_Pred)
    sigAvg = sum(Sigma2_Pred)/len(Sigma2_Pred)
    print("Prediction done.")
    # Now lets plot the Y_Train and Y_Pred
    print("Now plot the data.")
    time = np.array(range(ylength))
    pylab.plot(time, Y_Pred, '-b', label='Predicted')
    pylab.plot(time, Y_Test, ':k', label='Actual')
    pylab.xlabel('time axes')
    pylab.title("All Price in X, corr as: "+corr+", Avg Sigma_Pred: "+str(sigAvg))
    pylab.legend()
    pylab.savefig('all_'+ corr +'.png')
    pylab.show()

    # now calculate the root mean square error for our prediction
    error = 0
    for i in range(ylength):
        error += (Y_Pred[i] - Y_Test[i])*(Y_Pred[i] - Y_Test[i])
    error = math.sqrt(error/ylength)
    print("RMS Error: "+str(error))
