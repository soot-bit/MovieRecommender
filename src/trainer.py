from src.py.vanilla.data import DataLoader
import numpy as np

class Trainer:
    def __init__(self, dataset="ds100k", latent_d=5, lamda=0.001, tau=0.01, gamma=0.001):
        self.data = DataLoader(dataset)
        self.latent_d = latent_d
        self.lamda = lamda
        self.tau = tau
        self.gamma = gamma

        self.train_metrics = []

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


