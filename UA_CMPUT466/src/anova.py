import numpy as np
import matplotlib.pyplot as plt

#data
X_train = np.asarray([[0,1],[1,0],[1,2],[2,1],[1,1],[2,0],[2,2],[3,1]])
y_train = np.asarray([0]*4+[1]*4)

class LogitReg(Classifier):

    def __init__(self):
        self.weights = None

    def reset(self, parameters):
        self.weights = None
  
    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0  
        ### YOUR CODE HERE
        cost = -np.sum(y*np.log(1./(1+np.exp(-np.sum(theta*X, axis=1))))+\
                (1-y)*np.log(1.-1./(1+np.exp(-np.sum(theta*X, axis=1)))))
        #using regularizer 
        if self.params['regularizer'] == 'None':
            return cost
        else:
            return cost + self.params['regwgt']*self.regularizer[0](theta)
        
        ### END YOUR CODE

        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE
        grad = -(np.sum((utils.sigmoid(np.sum(theta*X,axis=1))-y)*X.T, axis=1)+\
                2*self.params['regwgt']*theta)
        ### END YOUR CODE

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        self.weights = np.zeros(Xtrain.shape[1],)

        batch_size = 1024
        ### YOUR CODE HERE
        for i in range(1000): #mini-batch gradient descent
            s = np.arange(Xtrain.shape[0])#shuffle the dataset
            np.random.shuffle(s)
            Xtrain = Xtrain[s]
            ytrain = ytrain[s]
            step = 0
            while True:
                if step*batch_size > Xtrain.shape[0]:#when there's not enough samples left for a batch
                    break

                Xbatch = Xtrain[step*batch_size:min((step+1)*batch_size,Xtrain.shape[0]), :]
                ybatch = ytrain[step*batch_size:min((step+1)*batch_size,Xtrain.shape[0])]
                grads = self.logit_cost_grad(self.weights, Xbatch, ybatch)
                self.weights += 0.001*grads
                step += 1

        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        probs = 1./(1.+np.exp(-np.sum(self.weights*Xtest,axis=1)))#sigmoid transfer
        ytest[probs>0.5] = 1

        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

