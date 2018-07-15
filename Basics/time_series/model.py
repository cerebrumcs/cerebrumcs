
'''
'''
import numpy as np

class HMM:
    

    def __init__(self, no_of_states, no_of_observations):
        self.nb_states = no_of_states
        self.nb_obs = no_of_observations
         
        self.states = list(range(self.nb_states))
        self.transition_probs = np.random.rand(self.nb_states, self.nb_states)
        self.initial_probs = np.random.rand(self.nb_states)
        self.emission_probs = np.random.rand(self.nb_states, self.nb_obs)
        

    def train(self, observations, train_epochs = 10):
        
        for s in range(train_epochs):
        
            self.forward_procedure(observations)
            self.backward_procedure(observations)
            
            self.expectation()
            self.maximization()
        


    
    def predict(self, observation):
        pass

    
    def expectation(self):
        pass 
    
    def maximization(self):
        pass
    
    
    def forward_procedure(self, observation):
        
        '''
            Calculates the probabilities that the 
            HMM produces an output sequence o_1,...,o_t and 
            ends in the state z_t: p(o_1,...,o_t, z_t|Theta), for all t.
        '''
        points_in_time = observation.shape
        alphas = np.zeros(shape=(points_in_time, self.nb_states))
        
        for t, o in zip(range(points_in_time), observation):
            b = self.emission_probs[:,o]
            
            if t == 0:
                alphas[t,:] = b * self.initial_probs
            else:                
                alphas[t,:] = b * np.matmul(alphas[t-1:], self.transition_probs)
                
        return alphas
    
    
    def backward_procedure(self, observation):
        '''
            Calculates the probablities that the HMM 
            is in state zt and will produce the output
            sequence o_t+1,...O_T in the remaining steps:
            p(o_t+1,...,O_T|z_t, Theta), for all t.
        '''
        points_in_time = observation.shape
        betas = np.zeros(shape=(points_in_time, self.nb_states))
        
        for t, o in zip(reversed(range(points_in_time)), observation):
            b = self.emission_probs[:,o]
            
            if t + 1 == points_in_time:
                betas[t,:] = 1
            else:
                betas[t,:] = b * np.matmul(betas[t+1,:], self.transition_probs)
                    
        return betas