
import time
import numpy as np
from tqdm import tqdm as tq
from utils import plot_loss, plot_rmse


class Trainor:
    """ Matrix factorisation using ALS algorithm """
    def __init__(self, data, latent_d=5, lamda = 0.001, tau = 0.01, gamma = 0.001):
        #Hyper_parameters
        self.lamda, self.tau, self.gamma = lamda, tau, gamma
        self.latent_d = latent_d
        self.user_data = data[0]
        self.item_data = data[1]
        self.test_data = data[2]
        self.user_matrix = np.random.normal(0, 1/np.sqrt(latent_d), size=(len(self.user_data), latent_d))
        self.item_matrix = np.random.normal(0, 1/np.sqrt(latent_d), size=(len(self.item_data), latent_d))
        self.user_bias = np.zeros((len(self.user_data)))
        self.item_bias = np.zeros((len(self.item_data)))
        #iloss list
        self.loss = []
        self.rmse = []
        # Test Error
        self.loss_test = []
        self.rmse_test = []
    

    def _update_user(self):
        
        for m in range(len(self.user_data)):

            if not self.user_data[m]:
                continue
            # Set the biases
            bm = 0
            user_counter  = 0
            
            for n, r in self.user_data[m]:
                inner = np.dot(self.user_matrix[m,:], self.item_matrix[n,:])
                bm += self.lamda*(r - (inner + self.item_bias[n]))
                user_counter += 1
                        
            self.user_bias[m] =  bm / (self.lamda * user_counter + self.gamma)
    
            # Set the trait vectors
            b_matrix = 0
            inv_matrix = 0
            
            for n, r in self.user_data[m]:
                
                b_matrix += self.lamda *self.item_matrix[n,:] *(r - ( self.user_bias[m] + self.item_bias[n]))                
                inv_matrix += self.lamda*(np.outer(self.item_matrix[n,:], self.item_matrix[n,:]))
                                
            inv_matrix += self.tau * np.identity(self.latent_d)                
         
            self.user_matrix[m,:] = np.dot(np.linalg.inv(inv_matrix), b_matrix)    
                
    def _update_item(self):
        
        for n in range(len(self.item_data)): 
            if not self.item_data[n]:
                continue
            # Set the biases
            bn = 0
            counter_bias = 0
            
            for m, r in self.item_data[n]:
                inner = np.dot(self.user_matrix[m,:], self.item_matrix[n,:])
                bn += self.lamda*(r - (inner + self.user_bias[m]))
                counter_bias += 1
                        
            self.item_bias[n] =  bn / (self.lamda * counter_bias + self.gamma)
    
            # Set the trait vectors
            b = 0 # A^(-1)b = ?
            outer = 0
            
            for m, r in self.item_data[n]:
                b += self.lamda *self.user_matrix[m,:]*(r - ( self.user_bias[m] + self.item_bias[n]))                
                outer += self.lamda*(np.outer(self.user_matrix[m,:], self.user_matrix[m,:]))
        
            outer += self.tau * np.identity(self.latent_d)              
            self.item_matrix[n,:] = np.dot(np.linalg.inv(outer), b) 
            
    def train(self, epoch=20, showplt = True):
        for epoch in tq(range(epoch)):
            self._update_user()
            self._update_item()

            #computing metrics
            l , r = self.compute_cost(data=self.user_data)
            l_test, r_test = self.compute_cost(data=self.test_data)
            print('epoch:', epoch + 1, '  rmse train:', r,' | rmse test:', r_test)
            self.loss.append(l)
            self.rmse.append(r)
            self.loss_test.append(l_test)
            self.rmse_test.append(r_test)
            
        if showplt:
            plot_loss(self.loss, self.loss_test )
            plot_rmse(self.rmse, self.rmse_test )


    def compute_cost(self, data):
        loss = 0
        rmse = 0
        count = 0
        for m in range(len(data)):
            for n, r in data[m]:
                error = (r - (np.dot(self.user_matrix[m,:], self.item_matrix[n,:]) + self.user_bias[m] + self.item_bias[n]))**2
                loss += self.lamda / 2 * error
                rmse += error
                count += 1

        loss += (self.gamma/2)*np.dot(self.item_bias,self.item_bias) + (self.gamma/2)*np.dot(self.user_bias,self.user_bias)
        
        for n in range(len(self.item_matrix)):
            loss += (self.tau/2)*np.dot(self.item_matrix[n,:], self.item_matrix[n,:])
            
        for m in range(len(self.user_matrix)):
            loss += (self.tau/2)*np.dot(self.user_matrix[m,:], self.user_matrix[m,:])

        return loss, np.sqrt(rmse/(count))


    def predictor(self, mov_title_dict, user_train_data, user_indx = 0 , weight = 0.2, dummy = False):
        if dummy:
            r_pred = [r for r in (np.inner(self.item_matrix[user_indx], self.item_matrix) + weight*self.item_bias) ] 
        else:
            r_pred = [r for r in (np.inner(self.user_matrix[user_indx], self.item_matrix) + weight*self.item_bias) ] 

        indexed_r = [ (m_idx, r) for m_idx,r in enumerate(r_pred)]  #index predictions
        
        # remove watched movies by user
        watched_movies = [t[0] for t in user_train_data[user_indx]] #mindx
        unwatched = [ indexed_r[i] for i in range(len(indexed_r)) if i not in watched_movies] 
        
        #sort and recommend
        recommend  = sorted(unwatched, key=lambda x: x[1], reverse=True)

        print("AI recommnds:........ ")
        for i , movie in enumerate(recommend[:10]):
            print( "\t", i+1, mov_title_dict[movie[0]], flush=True)
            time.sleep(0.5)

    def _reset(self):
        self.user_matrix = np.random.normal(0, 1/np.sqrt(self.latent_d), size=(len(self.user_data), self.latent_d))
        self.item_matrix = np.random.normal(0, 1/np.sqrt(self.latent_d), size=(len(self.item_data), self.latent_d))
        self.user_bias = np.zeros((len(self.user_data)))
        self.item_bias = np.zeros((len(self.item_data)))
        self.rmse_test = []
        self.rmse = []
