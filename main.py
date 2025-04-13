from src.data import SnapIndex as Indx
from build import cppEngine

def main():
    # Load and prepare data
    si = Indx(dataset="ml-latest-small")
    tensor = si.prepare_tensor()
    
    # Initialize and train model
    model = cppEngine.ALS(dim=64, reg=0.1, bias_reg=0.01, factor_reg=0.01)
    history = model.fit(tensor, iterations=20)
    
    # Show results
    print("\nTraining complete!")
    print(f"Final Test RMSE: {history[-1].test_rmse:.4f}")

if __name__ == "__main__":
    main()