from src.py.vanilla.data import DataLoader
import numpy as np

class Trainer:
    def __init__(self, dataset="ds100k", latent_d=5, lamda=0.001, tau=0.01, gamma=0.001):
        self.data = DataLoader(dataset)
        self.latent_d = latent_d
        self.lamda = lamda
        self.tau = tau
        self.gamma = gamma

        self.loss_history = []
        self.rmse_history = []
        self.loss_test_history = []
        self.rmse_test_history = []
        self._setup()
    
    def _setup(self):
        self.data.tt_split()

    def fit(self, model, epochs=20):
        self.model = model(self.data.user_train, self.data.movie_train, self.latent_d, self.lamda, self.tau, self.gamma)

        for epoch in range(epochs):
            self.model.update_U()
            self.model.update_V()
            l, r = self.calculate_loss(self.data.user_train)
            l_test, r_test = self.calculate_loss(self.data.user_test)
            self.loss_history.append(l)
            self.rmse_history.append(r)
            self.loss_test_history.append(l_test)
            self.rmse_test_history.append(r_test)
            print(f'Epoch: {epoch + 1}, RMSE Train: {r}, RMSE Test: {r_test}')


    def calculate_loss(self,usr_data):
        loss = self.l2_loss()
        rmse, count = self.calculate_rmse(usr_data)
        return loss, np.sqrt(rmse / count)

    def l2_loss(self):
        reg_loss = (self.gamma / 2) * (np.dot(self.model.item_bias, self.model.item_bias) + np.dot(self.model.user_bias, self.model.user_bias))

        for n in range(len(self.model.item_matrix)):
            reg_loss += (self.tau / 2) * np.dot(self.model.item_matrix[n, :], self.model.item_matrix[n, :])
        
        for m in range(len(self.model.user_matrix)):
            reg_loss += (self.tau / 2) * np.dot(self.model.user_matrix[m, :], self.model.user_matrix[m, :])

        return reg_loss

    def calculate_rmse(self, data):
        rmse = 0
        count = 0
        for user_index in range(len(data)):
            for item_index, rating in data[user_index]:
                error = (rating - (np.dot(self.user_matrix[user_index, :], self.item_matrix[item_index, :]) +
                                   self.user_bias[user_index] + self.item_bias[item_index])) ** 2
                rmse += error
                count += 1
        return rmse, count
