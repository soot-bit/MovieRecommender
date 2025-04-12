#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <chrono>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>

#include <thread>
#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

namespace py = pybind11;
namespace ind = indicators;

//------------------------------------------
//      Data Structure for Training
//------------------------------------------

struct snapTensor {
    std::vector<std::vector<std::tuple<int, float>>> train_user;
    std::vector<std::vector<std::tuple<int, float>>> train_item;
    std::vector<std::vector<std::tuple<int, float>>> test_user;
    std::vector<std::vector<std::tuple<int, float>>> test_item;

    void resize_users(int n_users) {
        train_user.resize(n_users);
        test_user.resize(n_users);
    }

    void resize_items(int n_items) {
        train_item.resize(n_items);
        test_item.resize(n_items);
    }

    void add_train(int user, int item, float rating) {
        train_user[user].emplace_back(item, rating);
        train_item[item].emplace_back(user, rating);
    }

    void add_test(int user, int item, float rating) {
        test_user[user].emplace_back(item, rating);
        test_item[item].emplace_back(user, rating);
    }
};

//------------------------------------------
//                  ALS
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
    float reg, bias_reg, factor_reg;
    Eigen::MatrixXf user_factors, item_factors;
    Eigen::VectorXf user_bias, item_bias;
    xt::xarray<float> user_factors_xt, item_factors_xt;

    void update_users(const snapTensor& data) {
        int n_users = data.train_user.size();
        for (int u = 0; u < n_users; ++u) {
            const auto& ratings = data.train_user[u];
            if (ratings.empty()) continue;

            int n_ratings = ratings.size();
            Eigen::VectorXf r(n_ratings);
            Eigen::MatrixXf V(n_ratings, dim);
            Eigen::VectorXf item_biases(n_ratings);

            for (int i = 0; i < n_ratings; ++i) {
                const auto& [item, rating] = ratings[i];
                r(i) = rating;
                V.row(i) = item_factors.row(item);
                item_biases(i) = item_bias[item];
            }

            Eigen::VectorXf pred = V * user_factors.row(u).transpose() +
                                    Eigen::VectorXf::Constant(n_ratings, user_bias[u]) +
                                    item_biases;
            Eigen::VectorXf residuals = r - pred;
            user_bias[u] = (reg * residuals.sum()) / (reg * n_ratings + bias_reg);

            Eigen::MatrixXf A = reg * V.transpose() * V + factor_reg * Eigen::MatrixXf::Identity(dim, dim);
            Eigen::VectorXf b = reg * V.transpose() * (r - Eigen::VectorXf::Constant(n_ratings, user_bias[u]) - item_biases);
            user_factors.row(u) = A.ldlt().solve(b).transpose();
        }
    }

    void update_items(const snapTensor& data) {
        int n_items = data.train_item.size();
        for (int i = 0; i < n_items; ++i) {
            const auto& ratings = data.train_item[i];
            if (ratings.empty()) continue;

            int n_ratings = ratings.size();
            Eigen::VectorXf r(n_ratings);
            Eigen::MatrixXf U(n_ratings, dim);
            Eigen::VectorXf user_biases(n_ratings);

            for (int j = 0; j < n_ratings; ++j) {
                const auto& [user, rating] = ratings[j];
                r(j) = rating;
                U.row(j) = user_factors.row(user);
                user_biases(j) = user_bias[user];
            }

            Eigen::VectorXf pred = U * item_factors.row(i).transpose() +
                                    Eigen::VectorXf::Constant(n_ratings, item_bias[i]) +
                                    user_biases;
            Eigen::VectorXf residuals = r - pred;
            item_bias[i] = (reg * residuals.sum()) / (reg * n_ratings + bias_reg);

            Eigen::MatrixXf A = reg * U.transpose() * U + factor_reg * Eigen::MatrixXf::Identity(dim, dim);
            Eigen::VectorXf b = reg * U.transpose() * (r - user_biases - Eigen::VectorXf::Constant(n_ratings, item_bias[i]));
            item_factors.row(i) = A.ldlt().solve(b).transpose();
        }
    }

    Metrics compute_metrics(const snapTensor& data) {
        float loss = 0.0f, train_rmse = 0.0f, test_rmse = 0.0f;
        int train_count = 0, test_count = 0;

        for (size_t u = 0; u < data.train_user.size(); ++u) {
            for (const auto& [i, r] : data.train_user[u]) {
                float pred = user_factors.row(u).dot(item_factors.row(i)) + user_bias[u] + item_bias[i];
                float error = r - pred;
                loss += error * error;
                train_count++;
            }
        }

        for (size_t u = 0; u < data.test_user.size(); ++u) {
            for (const auto& [i, r] : data.test_user[u]) {
                float pred = user_factors.row(u).dot(item_factors.row(i)) + user_bias[u] + item_bias[i];
                float error = r - pred;
                test_rmse += error * error;
                test_count++;
            }
        }

        loss += 0.5f * factor_reg * (user_factors.squaredNorm() + item_factors.squaredNorm());
        loss += 0.5f * bias_reg * (user_bias.squaredNorm() + item_bias.squaredNorm());

        return {loss, std::sqrt(loss / train_count), std::sqrt(test_rmse / test_count)};
    }

    void show_progress(int iter, int total, const Metrics& metrics, ind::ProgressBar& bar) {
        bar.set_option(ind::option::PostfixText{
            "Loss: " + std::to_string(metrics.loss) +
            " | Train RMSE: " + std::to_string(metrics.train_rmse) +
            " | Test RMSE: " + std::to_string(metrics.test_rmse)
        });
        bar.set_progress(100 * (iter + 1) / total);
    }

public:
    ALS(int dim, float reg, float bias_reg, float factor_reg)
    : dim(dim), reg(reg), bias_reg(bias_reg), factor_reg(factor_reg),
      user_factors(), item_factors(),  // Initialize Eigen members
      user_bias(), item_bias() {}

    std::vector<Metrics> fit(const snapTensor& data, int iterations = 10) {
        int n_users = data.train_user.size();
        int n_items = data.train_item.size();

        ind::show_console_cursor(false);
        ind::ProgressBar bar{
            ind::option::BarWidth{50},
            ind::option::Start{" ["},
            ind::option::Fill{"█"},
            ind::option::Lead{"█"},
            ind::option::Remainder{"-"},
            ind::option::End{"]"},
            ind::option::ForegroundColor{ind::Color::blue},
            ind::option::ShowPercentage{true},
            ind::option::ShowElapsedTime{true},
            ind::option::ShowRemainingTime{true},
            ind::option::PrefixText{"Training ALS"}
        };

        user_factors.resize(n_users, dim);
        item_factors.resize(n_items, dim);
        user_bias.resize(n_users);
        item_bias.resize(n_items);


        // init biases and U,V matrices
        float std_dev = 1.0f / std::sqrt(dim);
        xt::xarray<float> user_factors_xt = xt::random::randn<float>(
                                                {n_users, dim}, 0.0f, std_dev
                                            );
        user_factors = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 
                                  Eigen::Dynamic, Eigen::RowMajor>>(
                                    user_factors_xt.data(), n_users, dim);

        xt::xarray<float> item_factors_xt = xt::random::randn<float>(
                                                {n_items, dim}, 0.0f, std_dev
                                            );
        item_factors = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 
                                  Eigen::Dynamic, Eigen::RowMajor>>(
                                    item_factors_xt.data(),
                                    n_items,
                                    dim
        );

        user_bias.setZero(n_users);
        item_bias.setZero(n_items);

        std::vector<Metrics> history;
        for (int iter = 0; iter < iterations; ++iter) {
            update_users(data);
            update_items(data);

            auto metrics = compute_metrics(data);
            history.push_back(metrics);

            show_progress(iter, iterations, metrics, bar);
            std::cout << std::flush;
        }

        bar.mark_as_completed();
        ind::show_console_cursor(true);
        return history;
    }
};

PYBIND11_MODULE(cppEngine, m) {
    py::class_<snapTensor>(m, "snapTensor")
        .def(py::init<>())
        .def("resize_users", &snapTensor::resize_users)
        .def("resize_items", &snapTensor::resize_items)
        .def("add_train", &snapTensor::add_train)
        .def("add_test", &snapTensor::add_test);

    py::class_<ALS>(m, "ALS")
        .def(py::init<int, float, float, float>())
        .def("fit", &ALS::fit);

    py::class_<Metrics>(m, "Metrics")
        .def(py::init<float, float, float>())
        .def_readonly("loss", &Metrics::loss)
        .def_readonly("train_rmse", &Metrics::train_rmse)
        .def_readonly("test_rmse", &Metrics::test_rmse);
}
