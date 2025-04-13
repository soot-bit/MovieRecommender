from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def process(history):
    metrics = defaultdict(list)
    for metric in history:
        metrics["loss"].append(metric.loss)
        metrics["test_rmse"].append(metric.test_rmse)
        metrics["train_rmse"].append(metric.train_rmse)
    return metrics

def view(metrics, figsize=(12, 6)):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(metrics["train_rmse"], label="Train RMSE", color="blue")
    ax1.plot(metrics["test_rmse"], label="Test RMSE", color="orange")
    
    ax1.axhline(0.9525, label="Cinematch RMSE", linestyle="--", color="red")
    ax1.axhline(0.8567, label="Netflix Prize RMSE", linestyle="--", color="green")
    
    ax1.legend()
    ax1.set_title("RMSE")
    ax1.set_ylabel("RMSE")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(metrics["loss"], label="Loss", color="purple")
    
    ax2.legend()
    ax2.set_title("Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


class Trainer:
    def __init__(self):
        self.metrics = defaultdict(list)

    @staticmethod
    def fit(model, epochs=20):
        metrics = defaultdict(list)
        pbar = tqdm(range(epochs), desc="Training.. ", unit="epoch")
        for epoch in pbar:
            model.update_U()
            model.update_V()
            l, r = model.get_metrics(mode="train")
            _, r_test = model.get_metrics(mode="test")

            # Append metrics
            metrics["loss"].append(l)
            metrics["train_rmse"].append(r)
            metrics["test_rmse"].append(r_test)

            pbar.set_postfix({
                "Loss": f"{l:.4f}",
                "Train RMSE": f"{r:.4f}",
                "Test RMSE": f"{r_test:.4f}"
            })
        return metrics

