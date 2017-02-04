# Neural Networks Demystified
# Part 5: Numerical Gradient Checking
#
# Supporting code for short YouTube series on artificial neural networks.
#
# Stephen Welch
# @stephencwelch


## ----------------------- Part 1 ---------------------------- ##
import numpy as np

# X = (hours sleeping, hours studying), y = Score on test


#X = np.array([[0,0,1],
#            [0,1,1],
#            [1,0,1],
#            [1,1,1]])
#    
#y = np.array([[1],
#			[1],
#			[1],
#			[0]])

## ----------------------- Part 5 ---------------------------- ##

class Neural_Network(object):
    def __init__(self, Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 5
        self.scalar=0.01
        self.Lambda=Lambda
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
        
    def optimizef(self, X,y):
        epoch=200000
        ntime=10000
        ran=epoch/ntime
        cost_tr=np.zeros(ran)
        j=0
        for i in range(epoch):
            dJdW1, dJdW2 = self.costFunctionPrime(X,y)
            self.W1 = self.W1 - self.scalar*dJdW1
            self.W2 = self.W2 - self.scalar*dJdW2
            cost3 = self.costFunction(X, y)
            if(i%ntime==0):
                print("cost is: {}".format(cost3))
                cost_tr[j]=cost3
                j+=1

            
        return cost_tr

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 


        



 
def main():
    X = np.array(([3,5], [5,1], [10,2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)
    
#Training Data:
    trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
    trainY = np.array(([75], [82], [93], [70]), dtype=float)

#Testing Data:
    testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
    testY = np.array(([70], [89], [85], [75]), dtype=float)

#Normalize:
    trainX = trainX/np.amax(trainX, axis=0)
    trainY = trainY/100 #Max test score is 100

# Normalize
    X = X/np.amax(X, axis=0)
    y = y/100 #Max test score is 100
    
    NN = Neural_Network(Lambda=0.0001)

    numgrad = computeNumericalGradient(NN, trainX, trainY)

    grad = NN.computeGradients(trainX,trainY)

    out=np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)

    #print(out)
    #scalar=0.01
    cost_tr=NN.optimizef(trainX,trainY)
    print(cost_tr)
    out=NN.forward(trainX)
    print(out)
    print('\n')
    print(trainY)
    
    
    
    
           
    
if __name__ == "__main__":
    main()

    
    
        

