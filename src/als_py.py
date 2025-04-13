import numpy as np

class ALSvanilla:
    """ Matrix Factorisation using ALS algorithm """
    def __init__(self, data_indexing, latent_d=5, lambda_=0.001, tau=0.01, gamma=0.001):
        self.user_data = data_indexing.user_train
        self.item_data = data_indexing.user_test
        
        self.lamda = lambda_
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


class optALS:
    """ Optimized Matrix Factorisation using ALS """
    def __init__(self, data_indexing, dim, lambda_=0.01, tau=0.1, gamma=0.1 ):
        self.di = data_indexing
        self.lambda_ = lambda_
        self.tau = tau
        self.gamma = gamma
        self.latent_d = dim

        #float32 
        self.user_matrix = np.random.normal(0, 1/np.sqrt(self.latent_d), 
                           size=(len(self.di.user_train), self.latent_d)).astype(np.float32)
        self.item_matrix = np.random.normal(0, 1/np.sqrt(self.latent_d), 
                           size=(len(self.di.movie_train), self.latent_d)).astype(np.float32)
        self.user_bias = np.zeros(len(self.di.user_train), dtype=np.float32)
        self.item_bias = np.zeros(len(self.di.movie_train), dtype=np.float32)
    
    def update_U(self):

        for m, user_ratings in enumerate(self.di.user_train):
            if not user_ratings:
                continue
            
            # Get all item indices and ratings for this user
            indices, ratings = zip(*user_ratings)
            item_indices = np.array(indices)
            V_i = self.item_matrix[item_indices]
            ratings = np.array(ratings)

            # Phase 1: Update user bias 
            current_preds = V_i @ self.user_matrix[m] + self.item_bias[item_indices]
            residuals_bias = ratings - current_preds
            new_bias = (self.lambda_ * residuals_bias.sum()) / (self.lambda_ * len(ratings) + self.gamma)

            #  update bias
            self.user_bias[m] = new_bias

            # Phase 2: Update user vector
            residuals = ratings - (self.user_bias[m] + self.item_bias[item_indices])
            A =self.lambda_ * V_i.T @ V_i + self.tau * np.eye(self.latent_d)
            b =self.lambda_ * (residuals @ V_i)
            L = np.linalg.cholesky(A)
            self.user_matrix[m] = np.linalg.solve(L.T, np.linalg.solve(L, b))

    def update_V(self):

        for n, movie_ratings in enumerate(self.di.movie_train):
            if not movie_ratings:
                continue

            # Get all user indices and ratings for this item
            indices, ratings = zip(*movie_ratings)
            user_indices = np.array(indices)
            U_j = self.user_matrix[user_indices]
            ratings = np.array(ratings)

            # Phase 1: Update item bias (using CURRENT item vector)
            current_preds = U_j @ self.item_matrix[n] + self.user_bias[user_indices]
            residuals_bias = ratings - current_preds
            new_bias = (self.lambda_ * residuals_bias.sum()) / (self.lambda_ * len(ratings) + self.gamma)

            # Immediately update bias
            self.item_bias[n] = new_bias

            # Phase 2: Update item vector
            residuals = ratings - (self.user_bias[user_indices] + self.item_bias[n])
            A =self.lambda_ * U_j.T @ U_j + self.tau * np.eye(self.latent_d)
            b =self.lambda_ * (residuals @ U_j)
            L = np.linalg.cholesky(A)
            self.item_matrix[n] = np.linalg.solve(L.T, np.linalg.solve(L, b))


    def get_metrics(self, mode="train"):   
        data = getattr(self.di, f"user_{mode.lower()}")
        loss = self._l2_loss()
        rmse = self._get_rmse(data)
        return loss, rmse
    
    def _l2_loss(self):
        user_reg = 0.5 * self.tau * np.sum(self.user_matrix ** 2)
        item_reg = 0.5 * self.tau * np.sum(self.item_matrix ** 2)
        bias_reg = 0.5 * self.gamma* (np.sum(self.user_bias ** 2) + np.sum(self.item_bias ** 2))
        return user_reg + item_reg + bias_reg
    
    def _get_rmse(self, data):
        sum_errors = 0
        count = 0
        for user_index in range(len(data)):
            for item_index, rating in data[user_index]:
                error = (rating - (np.dot(self.user_matrix[user_index, :], self.item_matrix[item_index, :]) +
                                   self.user_bias[user_index] + self.item_bias[item_index])) ** 2
                sum_errors += error
                count += 1
        return np.sqrt(sum_errors / count).item()

    def predict(self, m, n):
        return np.dot( self.user_matrix[m, :], self.item_matrix[n, :]
                      ) + self.user_bias[m] + self.item_bias[n]