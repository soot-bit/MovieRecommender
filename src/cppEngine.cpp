#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <chrono>
#include <thread>

#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <random>

#include <cmath>
#include <tuple>
#include <omp.h>

#include <iostream>

namespace py = pybind11;

//------------------------------------------
//      a Dstructure used for training
//------------------------------------------

struct snapTensor {
    std::vector<std::vector<std::tuple<int, float>>> user_train;
    std::vector<std::vector<std::tuple<int, float>>> movie_train;
    std::vector<std::vector<std::tuple<int, float>>> user_test;
    std::vector<std::vector<std::tuple<int, float>>> movie_test;

    void reshape(int n_users, int n_items) {
        user_train.resize(n_users);
        user_test.resize(n_users);
        movie_train.resize(n_items);
        movie_test.resize(n_items);
    }

    void add_train(int user, int item, float rating) {
        user_train[user].emplace_back(item, rating);
        movie_train[item].emplace_back(user, rating);
    }

    void add_test(int user, int item, float rating) {
        user_test[user].emplace_back(item, rating);
        movie_test[item].emplace_back(user, rating);
    }
};

//------------------------------------------
//                  ALS Alg
//------------------------------------------

struct Metrics {
    float loss;
    float train_rmse;
    float test_rmse;
    Metrics(float l, float tr, float ts) : loss(l), train_rmse(tr), test_rmse(ts) {}
};

class ALS {
private:
    int dim;
    float lambda_, tau, gamma;
    Eigen::MatrixXf user_factors;
    Eigen::MatrixXf item_factors;
    Eigen::VectorXf user_bias;
    Eigen::VectorXf item_bias;

    void update_users(const snapTensor& data) {
        const auto& user_train = data.user_train;
        const size_t num_users = user_train.size();
        
        #pragma omp parallel for schedule(dynamic)
        for(size_t u = 0; u < num_users; ++u) {
            const auto& ratings = user_train[u];
            if(ratings.empty()) continue;
    
            const size_t n_ratings = ratings.size();
            Eigen::VectorXf r(n_ratings);
            Eigen::MatrixXf V(n_ratings, dim);
            Eigen::VectorXf item_b(n_ratings);
    
            // Populate r, V, and item_b
            for(size_t i = 0; i < n_ratings; ++i) {
                const auto& [item, rating] = ratings[i];
                r[i] = rating;
                V.row(i) = item_factors.row(item);
                item_b[i] = item_bias[item];
            }
    
            // Phase 1: Update user bias
            const Eigen::VectorXf current_pred = V * user_factors.row(u).transpose() 
                                               + item_b;
            const float new_bias = (lambda_ * (r - current_pred).sum()) 
                                 / (lambda_ * n_ratings + gamma);
            user_bias[u] = new_bias;
    
            // Phase 2: Update user factors 
            const Eigen::VectorXf residuals = r - Eigen::VectorXf::Constant(n_ratings, user_bias[u]) - item_b;
            const Eigen::MatrixXf A = lambda_ * V.transpose() * V + tau * Eigen::MatrixXf::Identity(dim, dim);
            const Eigen::VectorXf b = lambda_ * V.transpose() * residuals;
    
            //  Cholesky decomposition to stabilise inverstion
            Eigen::LLT<Eigen::MatrixXf> solver(A);
            user_factors.row(u) = solver.solve(b).transpose();
        }
    }
    
    void update_items(const snapTensor& data) {
        const auto& movie_train = data.movie_train;
        const size_t num_items = movie_train.size();
        
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < num_items; ++i) {
            const auto& ratings = movie_train[i];
            if(ratings.empty()) continue;
    
            const size_t n_ratings = ratings.size();
            Eigen::VectorXf r(n_ratings);
            Eigen::MatrixXf U(n_ratings, dim);
            Eigen::VectorXf user_b(n_ratings);
    
            // Populate r, U, and user_b
            for(size_t j = 0; j < n_ratings; ++j) {
                const auto& [user, rating] = ratings[j];
                r[j] = rating;
                U.row(j) = user_factors.row(user);
                user_b[j] = user_bias[user];
            }
    
            // Phase 1: Update item bias
            const Eigen::VectorXf current_pred = U * item_factors.row(i).transpose() 
                                               + user_b;
            const float new_bias = (lambda_ * (r - current_pred).sum()) 
                                 / (lambda_ * n_ratings + gamma);
            item_bias[i] = new_bias;
    
            // Phase 2: Update item factors (using updated bias)
            const Eigen::VectorXf residuals = r - user_b - Eigen::VectorXf::Constant(n_ratings, item_bias[i]);
            const Eigen::MatrixXf A = lambda_ * U.transpose() * U + tau * Eigen::MatrixXf::Identity(dim, dim);
            const Eigen::VectorXf b = lambda_ * U.transpose() * residuals;
    
            //  Cholesky decomposition inversion
            Eigen::LLT<Eigen::MatrixXf> solver(A);
            item_factors.row(i) = solver.solve(b).transpose();
        }
    }

    Metrics compute_metrics(const snapTensor& data) {
        float train_loss = 0.0f, test_loss = 0.0f;
        size_t train_count = 0, test_count = 0;

        //  metrics
        for(size_t u = 0; u < data.user_train.size(); ++u) {
            for(const auto& [i, r] : data.user_train[u]) {
                const float pred = user_factors.row(u).dot(item_factors.row(i)) + user_bias[u] + item_bias[i];
                const float error = r - pred;
                train_loss += error * error;
                train_count++;
            }
        }

        for(size_t u = 0; u < data.user_test.size(); ++u) {
            for(const auto& [i, r] : data.user_test[u]) {
                const float pred = user_factors.row(u).dot(item_factors.row(i)) + user_bias[u] + item_bias[i];
                const float error = r - pred;
                test_loss += error * error;
                test_count++;
            }
        }

        //loss
        const float reg_loss = 0.5f * tau * (user_factors.squaredNorm() + item_factors.squaredNorm())
                             + 0.5f * gamma * (user_bias.squaredNorm() + item_bias.squaredNorm());

        return Metrics(
            train_loss + reg_loss,
            std::sqrt(train_loss / train_count),
            std::sqrt(test_loss / test_count)
        );
    }

    void show_progress(int iter, int total, const Metrics& metrics) {
        const int width = 50;
        const int progress = (iter + 1) * width / total;
        
        std::cout << "\r"  // Return to start of line
                  << "Training [";
        
        // Progress bar
        for (int i = 0; i < progress; i++) std::cout << "=";
        for (int i = progress; i < width; i++) std::cout << " ";
        
        std::cout << "] " << (iter + 1) * 100 / total << "% "
                  << "Loss: " << metrics.loss
                  << " | Test: " << metrics.test_rmse
                  << std::flush;
        
        if (iter + 1 == total) std::cout << std::endl;
    }


public:
    ALS(int dim, float lambda_, float tau, float gamma)
        : dim(dim), lambda_(lambda_), tau(tau), gamma(gamma) {}

        void initialize(int num_users, int num_items) {
            const float std_dev = 1.0f / std::sqrt(dim);
            Eigen::Rand::P8_mt19937_64 urng{42};
            Eigen::Rand::NormalGen<float> norm_gen(0.0f, std_dev);
            

            user_factors = Eigen::MatrixXf::NullaryExpr(num_users, dim, 
                [&]() { return norm_gen(urng); });
                
            item_factors = Eigen::MatrixXf::NullaryExpr(num_items, dim,
                [&]() { return norm_gen(urng); });
    
            // init biases to zero
            user_bias = Eigen::VectorXf::Zero(num_users);
            item_bias = Eigen::VectorXf::Zero(num_items);
        }

        std::vector<Metrics> fit(const snapTensor& data, int epochs = 10) {
            initialize(data.user_train.size(), data.movie_train.size());
            std::vector<Metrics> history;
    
            // init for Python output
            std::cout << std::endl;
    
            for(int iter = 0; iter < epochs; ++iter) {
                update_items(data);
                update_users(data);
                Metrics metrics = compute_metrics(data);
                history.push_back(metrics);
                show_progress(iter, epochs, metrics);
            }
    
            return history;
        }
};

PYBIND11_MODULE(cppEngine, m) {
    py::class_<snapTensor>(m, "snapTensor")
        .def(py::init<>())
        .def("reshape", &snapTensor::reshape)
        .def("add_train", &snapTensor::add_train)
        .def("add_test", &snapTensor::add_test)
        .def_readwrite("user_train", &snapTensor::user_train)
        .def_readwrite("movie_train", &snapTensor::movie_train)
        .def_readwrite("user_test", &snapTensor::user_test)
        .def_readwrite("movie_test", &snapTensor::movie_test);

    py::class_<ALS>(m, "ALS")
        .def(py::init<int, float, float, float>(),
            py::arg("dim"),
            py::arg("lambda_"),
            py::arg("tau"),
            py::arg("gamma")
        )
        .def("fit", &ALS::fit,
            py::arg("data").noconvert(),
            py::arg("epochs") = 10
        );

    py::class_<Metrics>(m, "Metrics")
        .def(py::init<float, float, float>())
        .def_readonly("loss", &Metrics::loss)
        .def_readonly("train_rmse", &Metrics::train_rmse)
        .def_readonly("test_rmse", &Metrics::test_rmse);
}