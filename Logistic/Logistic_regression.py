import numpy as np 

import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1./(1+np.exp(-x))


class Logistic:
    def __init__(self,dim=3):
        self.dim=dim 
        self.weight=np.random.random((self.dim,))

    def fit(self,X,y,eta=0.01,epochs=100000):
        N  = X.shape[0]
        #mixed_index = np.random.permutation(N)
        epoch = 0
        prev_weight = self.weight
        
        while epoch < epochs:
            loss_grad = self._loss_grad(X,y)
            loss=self._loss(X,y)
            
            self.weight=self.weight - eta * loss_grad
            print("""
                epoch {}:
                loss:{}
                grad_loss:{} 
            """.format(epoch,loss,loss_grad))
          
            if np.linalg.norm((self.weight-prev_weight).T) < 1e-3 :
                
                break
            #print(self.weight)
            prev_weight=self.weight
            epoch += 1
        return self.weight
    
    
    def _loss(self,X,y):
        N = X.shape[0]
        total_loss = 0
        for i in range(N):
            x_i = X[i]
            y_i =  y[i]
            pred = self._predict(x_i)
            individual_loss = y_i*np.log(pred) + (1-y_i)*np.log(1-pred)
            total_loss += individual_loss
        return -total_loss/N
    
    def _predict(self,x):
        return sigmoid( x.dot( self.weight ))
    

    def _loss_grad(self,X,y):
        N = X.shape[0]
        #print(N)
        sum = np.zeros_like(self.weight)
        for i in range(N):
        
            sum += (y[i] - self._predict(X[i])) * X[i].T
        return -sum/N


if __name__ == "__main__" :
    means = [[2,2], [6, 2]]
    cov = [[.3, .1], [.1, .3]]
    N=500
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X = np.concatenate((X0, X1), axis = 0)
    y = np.concatenate((np.ones((N, 1)), np.zeros((N, 1))), axis=0)
    X = np.concatenate((X, np.ones((2*N, 1))), axis = 1)
    # print(X)
    # print(y)
    logistic=Logistic()   
    weight=logistic.fit(X, y)
    
    # print(weight)
    # print(pla.predict(X))
    plt.figure(figsize = (5, 3))
    plt.scatter(X0[:,0], X0[:,1], color = 'red')
    plt.scatter(X1[:,0], X1[:,1], color = 'blue')
    plt.scatter(means[0][0], means[0][1], s = 40, color ='yellow')
    plt.scatter(means[1][0], means[1][1], s = 40, color ='green')
    x_axis = np.array([0, 10])
    plt.plot(x_axis, -(weight[0]*x_axis + weight[2])/(weight[1]+0.0001))
    #print(weight)
    #plt.plot(x_axis, -(weight_init[0]*x_axis + weight_init[2])/weight_init[1], color='black')
    plt.show()
