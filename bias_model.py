
import numpy as np
from utils import plot_loss, plot_rmse

class BiasModel:
    """ Matrix factorisation using ALS algorithm for Biases """
    def __init__(self, data, , lamda = 0.001, tau = 0.5, gamma = 0.001):
        #Hyper_parameters
        self.lamda, self.tau, self.gamma = lamda, tau, gamma
        self.user_data = data[0]
        self.item_data = data[1]
        self.test_data = data[2]

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
                bm += self.lamda*(r - (inner + self.item_bias[n]))
                user_counter += 1
                        
            self.user_bias[m] =  bm / (self.lamda * user_counter + self.gamma)

                
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
    

    def train(self, epoch=10):
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
            self.item_bias = np.zeros((len(self.item_data)))
            self.rmse_test.append(r_test)

        plot_loss(self.loss, self.loss_test )
        plot_rmse(self.rmse, self.rmse_test )


    def compute_cost(self, data):
        loss = 0
        rmse = 0
        count = 0
        for m in range(len(data)):
            for n, r in data[m]:
                error = (r - ( self.user_bias[m] + self.item_bias[n]))**2
                loss += self.lamda / 2 * error
                rmse += error
                count += 1

        loss += (self.gamma/2)*np.dot(self.item_bias,self.item_bias) + (self.gamma/2)*np.dot(self.user_bias,self.user_bias)

        return loss, np.sqrt(rmse/(count))