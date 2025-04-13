from src.data import DataIndx
import cppEngine
from src.als_py import optALS
import argparse
from src.utils import *

def main():
    parser = argparse.ArgumentParser(description="ALS Recommender System")
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['ml-latest-small', 'ml-latest'],
                      help='Dataset to use for training')
    parser.add_argument('--flash', action='store_true',
                      help='Use C++ implementation')
    parser.add_argument('--plot', action='store_true',
                      help='plot train metrics?')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    args = parser.parse_args()


    di = DataIndx(dataset=args.dataset) 
    di.tt_split()
    snap_t = di.snap_tensor

    if args.flash:
        model = cppEngine.ALS(dim=10, lambda_=0.01, tau=0.1, gamma=0.1)
        history = model.fit(snap_t, epochs=args.epochs)
        if args.plot:
            view((process(history)))
    else:
        print("🐍")
        model = optALS(snap_t, dim=10, lambda_=0.1, tau=0.01, gamma=0.01 ) 
        metrics = Trainer.fit(model, epochs=args.epochs)
        if args.plot:
            view(metrics)


if __name__ == "__main__":
    main()
