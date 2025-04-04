import pathlib

import numpy as np
import pandas as pd
import torch
from pandas.testing import assert_frame_equal

path = pathlib.Path().resolve()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def test_complete_training():
    from ska.model import SKAModel
    from ska.utils import inputs, save_metric_csv

    # Training parameters
    model = SKAModel()

    learning_rate = 0.01

    # SKA training over multiple forward steps
    total_entropy = 0
    step_count = 0

    # Initialize tensors for first step
    model.initialize_tensors(inputs.size(0))

    # Process K forward steps (without backpropagation)
    for k in range(model.K):
        outputs = model.forward(inputs)
        # Store mean output distribution for the final layer
        model.output_history.append(
            outputs.mean(dim=0).detach().cpu().numpy()
        )  # [10] vector
        if k > 0:  # Compute entropy after first step
            batch_entropy = model.calculate_entropy()
            model.ska_update(inputs, learning_rate)
            total_entropy += batch_entropy
            step_count += 1
            print(f"Step: {k}, Total Steps: {step_count}, Entropy: {batch_entropy:.4f}")
            # Visualize Frobenius norm heatmap
            # After weight updates, compute and store weight Frobenius norms
            for l in range(len(model.layer_sizes)):  # noqa: E741
                weight_norm = torch.norm(model.weights[l], p="fro")
                model.weight_frobenius_history[l].append(weight_norm.item())

        # Update previous decision and knowledge tensors
        model.D_prev = [d.clone().detach() if d is not None else None for d in model.D]
        model.Z_prev = [z.clone().detach() if z is not None else None for z in model.Z]

    layers = len(model.layer_sizes)
    # Final statistics
    save_metric_csv(model.entropy_history, "test_entropy_history.csv", layers)
    save_metric_csv(model.cosine_history, "test_cosine_history.csv", layers)
    save_metric_csv(model.frobenius_history, "test_frobenius_history.csv", layers)
    save_metric_csv(
        model.weight_frobenius_history, "test_weight_frobenius_history.csv", layers
    )
    save_metric_csv(model.net_history, "test_tensor_net_history.csv", layers)

    files = [
        "entropy_history.csv",
        "cosine_history.csv",
        "frobenius_history.csv",
        "weight_frobenius_history.csv",
        "tensor_net_history.csv",
    ]

    for file in files:
        original_df = pd.read_csv(f"{path}\\tests\\{file}")
        test_df = pd.read_csv(f"{path}\\saved\\test_{file}")
        assert_frame_equal(original_df, test_df)
