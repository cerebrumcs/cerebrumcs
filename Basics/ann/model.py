'''
Created on 02.05.2018
'''

import numpy as np

class MLP(object):
        
    def __init__(self, input_dim, layer, output_func = "identity", loss_func = "lse"):
        '''
        layer : the definition of the layers. For example if layer=(2,3,1) a model with 3 layers is created, 
            whereby the first layer is build by 2 units, the second layer is build by 3 units and the third layer is
            build by 1 unit. 
        input_dim: dimension of the input data 
        '''
        
        output_dim = 1
        
        self.layer = layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weights, self.biases = MLP.init(self.input_dim, self.layer)
        self.output_func = output_func
        self.loss_func = loss_func
    
    @staticmethod
    def init(input_dim, layer):
        
        weights = []
        biases  = []
        output_dim = 1
        
        layer = (input_dim,) + layer + (output_dim,)
        
        ## weight initialization for each layer
        for i in range(1,len(layer)):
            previous_layer_units = layer[i-1]
            current_layer_units  = layer[i]            
            # one row per unit
            weights.append(np.random.rand(current_layer_units, previous_layer_units))
            biases.append(np.random.rand(current_layer_units))
        return weights, biases

    @staticmethod
    def z(A):
        #z_out = 1/(1 + np.exp(-A))
        z_out = np.tanh(A)
        return z_out         
    
    @staticmethod
    def a(x, W, B):
        a_out = np.matmul(W, x) + B                                                                              
        return a_out
           
    @staticmethod                   
    def forward_pass(L, x, output_func = "identity"):
        '''
            L : Layer
            x : Input Value (Sample)        
        '''        
        
        l = 0 
        A, Z= [], [x]
                
        # for each layer calculate the layer outputs and store them to use during backprop
        for l in range(len(L)):
            W, B = L[l]                                    
            A += [MLP.a(Z[-1], W, B)]
            
            if l < len(L):
                # hidden layer
                Z += [MLP.z(A[l])]            
            else:
                # output layer 
                if output_func is "identity":
                    Z += [A[l]]
                elif output_func is "logistic":
                    Z += [MLP.z(A[l])]    

        return A, Z
    
    
    @staticmethod
    def backward_pass(y, p, L, Z):
        '''
            y : Target Value (Label)
            p : Prediction
            L : the layer
            Z : A list of all Z values (activation outputs of the hidden layers)
        '''
        # for each layer l we want to                    
        # 1. calculate the gradient dE/dw 
        # 2. calculate the gradient dE/db_j
        
        # the derivatives wrt. the weights of an upper layer
        # can be enrolled by applying chain rule repeatedly.
        # thus, the derivatives are obtained by using the already computed
        # derivatives of the lower layers.
        
        # L contains all weighted layer. This are the hidden layers and one output layer.
        no_of_hidden_layers = len(L) - 1         

        # lets start with the output layer 
        # for which the activation is assumed to be the identity.
        # so, the calculation the gradient of E wrt. w,b in the output layer is straightforward
        da_dw = Z[-2]
        da_db = 1
        dp_da = 1
        dE_dp = p - y
        dE_da = dE_dp * dp_da 
                
        dE_dw = dE_da * da_dw        
        dE_db = dE_da * da_db

        # collecting all gradients is required to calculate the dataset global value
        dL = [(dE_dw, dE_db)] 

        # continue with the hidden layers - from last to first
        hidden_layer_idx  = reversed(range(no_of_hidden_layers))        
        for l in hidden_layer_idx:
            #TODO: just for debugging 
            # l = next(hidden_layer_idx)
            lz = l + 1 # idx of layer l in the Z-Array
            
            W, B = 0,1                  
            lm = Z[lz-1].shape[0] # previous layer size
            ln = Z[lz].shape[0]   # layer size
                        
            # da^(l+1)/dz^(l)
            # The vector A for an Layer is given by A(Z,W) = (a1,...,aN)  = (W1*Z,...,WN*Z). 
            # Thus, the differential is given by the jacobi matrix
            # (da1/dz1 ... da1/dzM) = (w11 ... wM1) 
            # (da2/dz1 ... da2/dzM) = (w12 ... wM2)
            # (        ...        ) = (    ...    )
            # (daN/dz1 ... daN/dzM) = (W1N ... WMN)                
            da_dz = L[l + 1][W]
            
            # dz^(l)/da^(l) 
            # For a layer with output Z=(z1,...,zN) the differential wrt. A=(a1,...,aN)
            # dZ / dA is given by the jacobi matrix
            # (dz1/da1 ... dz1/daN) = (z1(1 - z1),0,0,...,0)
            # (dz2/da1 ... dz2/daN) = (0,z2(1 - z2),0,...,0)
            # (        ...        ) = ()
            # (dzN/da1 ... dzN/daN) = (0,0,0,...,zN(1 - zN))
            dz_da = np.zeros((ln, ln))    
            #np.fill_diagonal(dz_da, Z[lz] * (1 - Z[lz]))
            np.fill_diagonal(dz_da, (1 - np.square(Z[lz])))
            
            # da^(l)/dw^(l) 
            # The vector A for an Layer is given by A(Z,W) = (a1,...,aN)  = (W1*Z,...,WN*Z). 
            # Thus, the differential is given by the jacobi matrix
            # (da1/dW11 ... da1/dWM1,...,da1/dW1N ... da1/dWMN) = (z1,...,zM,0,...,0,....,0,...,0)
            # (da2/dW11 ... da2/dWM1,...,da2/dW1N ... da2/dWMN) = (0,...,0,z1,...,zM,....,0,...,0)
            # (         ...                                   ) = (................................) 
            # (daN/dW11 ... daN/dWM1,...,daN/dW1N ... daN/dWMN) = (0,...,0,....,0,...,0,z1,...,zN)
            da_dw  = np.zeros((ln,lm*ln))                             
            for ll in range(ln):
                da_dw[ll,lm*ll:lm*ll+lm] = Z[lz-1]                

            # da^(l)/db^(l) 
            # (da1/db1 ... da1/dbN) = (1,0,...,0) 
            # (da2/db1 ... da2/dbN) = (0,1,...,0)
            # (        ...        ) = (    ...  )
            # (daN/db1 ... daN/dbN) = (0,0,...,1) 
            da_db  = np.zeros((ln,ln))                             
            np.fill_diagonal(da_db, 1)  

            # dE/dz^(l) = dE/da^(l+1) * da^(l+1)/dz^(l)                                            
            dE_dz = np.matmul(dE_da, da_dz) 
            # dE/da^(l) = dE/dz^(l) * dz^(l)/da^(l)
            dE_da = np.matmul(dE_dz, dz_da)                
            # dE/dw^(l) = dE/da^(l) * da^(l)/dw^(l)
            dE_dw = np.matmul(dE_da, da_dw)
            # dE/db^(l) = dE/da^(l) * da^(l)/db^(l)
            dE_db = np.matmul(dE_da, da_db)
            
            # reshape and collect
            dL += [(dE_dw.reshape(L[l][W].shape), dE_db.reshape(L[l][B].shape))]
                        
        return list(reversed(dL))
   
    @staticmethod
    def update_weights(L, dL, lr):
        L = np.array(L) - lr * np.array(dL)
        return L
    
    
    def fit(self, X, Y, epochs, learning_rate):
        for e in range(epochs):
            loss = 0
            L  = np.array(list(zip(self.weights, self.biases)))        
            dL = None
            for x,y in zip(X,Y):
                #TODO: just for debugging
                #x,y = next(xy)
                A, Z = MLP.forward_pass(L, x, "logistic")
                p = Z[-1]
                loss += (p - y)**2
                if dL is None:
                    dL = np.array(MLP.backward_pass(y, p, L, Z))
                else:
                    dL += np.array(MLP.backward_pass(y, p, L, Z))

            print("Finished epoch = ", e)
            print("Loss = ", loss)    
                
            L = MLP.update_weights(L, dL, learning_rate)
            W, B = 0, 1
            self.biases = L[:,B]
            self.weights = L[:,W]
            
    
    def predict(self, data):
        P = []
        L  = np.array(list(zip(self.weights, self.biases)))
        for x in data:            
            A, Z = self.forward_pass(L, x)
            P += [Z[-1]]
        return P
  
        
def main():
    from matplotlib import pyplot
    from sklearn.neural_network import MLPRegressor
    
    X = np.random.rand(100,1) * np.pi
    Y = np.sin(X)
    X = (X - np.mean(X)) / np.std(X)
    mlp = MLP(1,(20,5))
    mlp.fit(X, Y, 1000, 0.001)
    P = mlp.predict(X)
  
    pyplot.scatter(X,Y)
    pyplot.scatter(X,P)


    mlpc = MLPRegressor(hidden_layer_sizes=(10,50,5), activation="tanh", max_iter=1000)
    mlpc.fit(X, Y)
    PP = mlpc.predict(X)
    pyplot.scatter(X,Y)
    pyplot.scatter(X,PP)
