import pathlib

import pandas as pd
import torch

path = pathlib.Path(__file__).parent.parent.resolve()  #


### **Function to Save Data as CSV**
# Define the save_metric_csv function OUTSIDE the class
def save_metric_csv(metric_data, filename, layers):
    """Saves a 2D metric (list of lists) to a CSV file with layers as rows and correct step count."""
    actual_steps = min(len(layer) for layer in metric_data)  # Ensure correct step count
    df = pd.DataFrame(
        metric_data,
        index=[f"Layer {i+1}" for i in range(layers)],
        columns=[f"K={j+1}" for j in range(actual_steps)],
    )
    df.to_csv(f"{path}/saved/{filename}")
    print(f"Saved {filename} with {actual_steps} steps")


# Load the pre-saved MNIST subset (100 samples per class)
mnist_subset = torch.load(str(path / "data" / "mnist_subset_100_per_class.pt"))
images = torch.stack([item[0] for item in mnist_subset])  # Shape: [1000, 1, 28, 28]
labels = torch.tensor([item[1] for item in mnist_subset])

# Prepare the dataset (single batch for SKA forward learning)
inputs = images  # No mini-batches, full dataset used for forward-only updates
