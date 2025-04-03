import numpy as np
import torch

from ska.model import SKAModel
from ska.utils import inputs

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def test_single_pass_training():
    # Training parameters
    model = SKAModel()

    initial_entropy = model.entropy_history.copy()
    initial_frobenius = model.frobenius_history.copy()
    initial_weight = model.weight_frobenius_history.copy()
    initial_cosine = model.cosine_history.copy()
    initial_output = model.output_history.copy()
    initial_net_history = model.net_history.copy()

    model.K = 2
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

    # Final statistics

    assert initial_entropy != model.entropy_history
    assert initial_frobenius != model.frobenius_history
    assert initial_weight != model.weight_frobenius_history
    assert initial_cosine != model.cosine_history
    assert initial_output != model.output_history
    assert initial_net_history != model.net_history
