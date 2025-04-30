from src.data import DataIndx
import cppEngine
from src.als_py import optALS
import argparse
from src.utils import *
import optuna
import time
from rich.console import Console
console = Console()

def main():
    parser = argparse.ArgumentParser(description="ALS Recommender System")
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['ml-latest-small', 'ml-latest'],
                      help='Dataset to use for training')
    parser.add_argument('--flash', action='store_true',
                      help='Use C++ implementation')
    parser.add_argument('--plot', action='store_true',
                      help='plot train metrics?')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs')
    
    parser.add_argument('--lambda_', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--tune', action='store_true',
                      help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--trials', type=int, default=2,
                      help='Number of Optuna trials')
    
    args = parser.parse_args()




    di = DataIndx(dataset=args.dataset) 
    snap_t = di.snap_tensor

    if args.tune:
        console.rule("[!] Optuna tuning")
        def objective(trial):
            
            lambda_ = trial.suggest_float('lambda_', 1e-5, 1.0, log=True)
            tau = trial.suggest_float('tau', 1e-5, 1.0, log=True)
            gamma = trial.suggest_float('gamma', 1e-5, 1.0, log=True)

            if args.flash:
                # C++  tuning
                model = cppEngine.ALS(dim=10, lambda_=lambda_, tau=tau, gamma=gamma)
                history = model.fit(snap_t, epochs=args.epochs)
                metrics = process(history)
                test_rmse = metrics["test_rmse"][-1]
            else:
                # Python  tuning
                model = optALS(snap_t, dim=10, lambda_=lambda_, tau=tau, gamma=gamma)
                metrics = Trainer.fit(model, epochs=args.epochs)
                test_rmse = metrics['test_rmse'][-1]

            return test_rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=args.trials)

        print(f"Best trial:")
        print(f"  Value: {study.best_trial.value}")
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")



    if args.flash:
        s = time.time()
        print()
        console.rule("[⚡] Training")
        model = cppEngine.ALS(dim=10, lambda_=0.01, tau=0.1, gamma=0.1)
        history = model.fit(snap_t, epochs=args.epochs)
        e = time.time()
        console.rule(f"[⌛]: {e-s:.2f} sec")
        if args.plot:
            view((process(history)))

    else:  
        print()
        console.rule("[🐍] Training")
        s = time.time()
        model = optALS(snap_t, dim=10, lambda_=5.47e-5, tau=0.00159, gamma=5.8e-5) 
        metrics = Trainer.fit(model, epochs=args.epochs)
        e = time.time()
        console.rule(f"[⌛]: {e-s:.2f} sec")
        if args.plot:
            view(metrics)


if __name__ == "__main__":
    main()
