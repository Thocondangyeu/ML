import numpy as np 

import matplotlib.pyplot as plt 

class Perceptron:
    def __init__(self,dimension=3):
        self.dimension=dimension
        self.weight=np.zeros((dimension,))

    def fit(self,X,y,eta=0.001):
        N=X.shape[0]
        num_epoch=0

        while True:
            mixed_index=np.random.permutation(N)
            for i in mixed_index:
                
                if self._predict(X[i])!=y[i]:
                    
                    self.weight-=eta*self._loss_grad(X[i],y[i])

            
            num_epoch+=1
            num_fallen=self._number_fallen(X,y)

            print("""
Epoch {}:
loss:{}
num fallen:{}
            """.format(num_epoch,self._loss(X,y),num_fallen))
            if num_fallen == 0:
                break
            if num_epoch==10000:
                return self.weight
                        


            
        return self.weight


    
    
    def _predict(self,x):
        if x.dot(self.weight)>=0:
            
            return 1
        else: 
            return -1


    def _loss(self,X,y):
        return -np.sum(X.dot(self.weight)*y)
    
    
    def _loss_grad(self,x,y):
        return (-x*y)


    def _number_fallen(self,X,y):
        """
            X: Matrix N*d with : N is the  number of data points, n is the dimension of data
            y : matrix 
        """
        count=0
        for i in range(X.shape[0]):
        
            res=self._predict(X[i])
            if res != y[i]:
                count+=1
        return count


    
if __name__=="__main__":
    means = [[2,2], [6, 2]]
    cov = [[.3, .1], [.1, .3]]
    N=100
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X = np.concatenate((X0, X1), axis = 0)
    y = np.concatenate((np.ones((N, 1)), -1*np.ones((N, 1))), axis=0)
    X = np.concatenate((X, np.ones((2*N, 1))), axis = 1)
    # print(X)
    # print(y)
    pla = Perceptron()
   
    weight=pla.fit(X, y)
    
    # print(weight)
    # print(pla.predict(X))
    plt.figure(figsize = (5, 3))
    plt.scatter(X0[:,0], X0[:,1], color = 'red')
    plt.scatter(X1[:,0], X1[:,1], color = 'blue')
    plt.scatter(means[0][0], means[0][1], s = 40, color ='yellow')
    plt.scatter(means[1][0], means[1][1], s = 40, color ='green')
    x_axis = np.array([0, 10])
    plt.plot(x_axis, -(weight[0]*x_axis + weight[2])/weight[1])
    #print(weight)
    #plt.plot(x_axis, -(weight_init[0]*x_axis + weight_init[2])/weight_init[1], color='black')
    plt.show()
