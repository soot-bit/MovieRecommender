import numpy as np

class ALSModel:
    """ Matrix Factorisation using ALS algorithm """
    def __init__(self, user_data, item_data, latent_d=5, lamda=0.001, tau=0.01, gamma=0.001):
        self.user_data = user_data
        self.item_data = item_data
        
        self.lamda = lamda
        self.tau = tau
        self.gamma = gamma
        self.latent_d = latent_d


        # Initialise latent matrices and biases
        self.user_matrix = np.random.normal(0, 1/np.sqrt(latent_d), size=(len(self.user_data), latent_d))
        self.item_matrix = np.random.normal(0, 1/np.sqrt(latent_d), size=(len(self.item_data), latent_d))
        self.user_bias = np.zeros((len(self.user_data)))
        self.item_bias = np.zeros((len(self.item_data)))

    def update_U(self):
        for m in range(len(self.user_data)):
            if not self.user_data[m]:
                continue

            bm = 0
            user_counter = 0
            for n, r in self.user_data[m]:
                inner = np.dot(self.user_matrix[m, :], self.item_matrix[n, :])
                bm += self.lamda * (r - (inner + self.item_bias[n]))
                user_counter += 1

            self.user_bias[m] = bm / (self.lamda * user_counter + self.gamma)

            b_matrix = 0
            inv_matrix = 0
            for n, r in self.user_data[m]:
                b_matrix += self.lamda * self.item_matrix[n, :] * (r - (self.user_bias[m] + self.item_bias[n]))
                inv_matrix += self.lamda * (np.outer(self.item_matrix[n, :], self.item_matrix[n, :]))

            inv_matrix += self.tau * np.identity(self.latent_d)
            self.user_matrix[m, :] = np.dot(np.linalg.inv(inv_matrix), b_matrix)

    def update_V(self):
        for n in range(len(self.item_data)):
            if not self.item_data[n]:
                continue

            bn = 0
            counter_bias = 0
            for m, r in self.item_data[n]:
                inner = np.dot(self.user_matrix[m, :], self.item_matrix[n, :])
                bn += self.lamda * (r - (inner + self.user_bias[m]))
                counter_bias += 1

            self.item_bias[n] = bn / (self.lamda * counter_bias + self.gamma)

            b = 0
            outer = 0
            for m, r in self.item_data[n]:
                b += self.lamda * self.user_matrix[m, :] * (r - (self.user_bias[m] + self.item_bias[n]))
                outer += self.lamda * (np.outer(self.user_matrix[m, :], self.user_matrix[m, :]))

            outer += self.tau * np.identity(self.latent_d)
            self.item_matrix[n, :] = np.dot(np.linalg.inv(outer), b)

    def predict(self, m, n):
        return np.dot(self.user_matrix[m, :], self.item_matrix[n, :]) + self.user_bias[m] + self.item_bias[n]


