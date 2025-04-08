import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ska_path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(ska_path))  # Convert to string to ensure compatibility
import ska as ska  # noqa: E402
from ska.model import SKAModel  # noqa: E402
from ska.utils import inputs, save_metric_csv  # noqa: E402

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Training parameters
model = SKAModel()
learning_rate = 0.01

# SKA training over multiple forward steps
total_entropy = 0
step_count = 0
start_time = time.time()

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
        model.visualize_entropy_heatmap(step_count)
        model.visualize_cosine_heatmap(step_count)
        # Visualize Frobenius norm heatmap
        model.visualize_frobenius_heatmap(step_count)
        # After weight updates, compute and store weight Frobenius norms
        for l in range(len(model.layer_sizes)):  # noqa: E741
            weight_norm = torch.norm(model.weights[l], p="fro")
            model.weight_frobenius_history[l].append(weight_norm.item())
        model.visualize_weight_frobenius_heatmap(step_count)
        model.visualize_net_heatmap(step_count)  # Visualize per-step Tensor Net
        model.visualize_entropy_vs_frobenius(step_count)

    # Update previous decision and knowledge tensors
    model.D_prev = [d.clone().detach() if d is not None else None for d in model.D]
    model.Z_prev = [z.clone().detach() if z is not None else None for z in model.Z]

# Final statistics
total_time = time.time() - start_time
avg_entropy = total_entropy / step_count if step_count > 0 else 0
print(
    f"Training Complete: Avg Entropy={avg_entropy:.4f}, Steps={step_count}, Time={total_time:.2f}s"
)
print(
    f"Tensor Net Total per layer: {[f'Layer {i+1}: {tn:.4f}' for i, tn in enumerate(model.tensor_net_total)]}"
)

# Plot historical evolution for all metrics
plt.figure(figsize=(8, 6))
plt.plot(np.array(model.entropy_history).T)  # Entropy
plt.title("Entropy Evolution Across Layers (Single Pass)")
plt.xlabel("Step Index K")
plt.ylabel("Entropy")
plt.legend([f"Layer {i+1}" for i in range(len(model.layer_sizes))])
plt.grid(True)
plt.savefig(f"{ska_path}/figures/entropy/entropy_history_single_pass.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(np.array(model.cosine_history).T)  # Cosine
plt.title("Cos(\u03B8) Alignment Evolution Across Layers (Single Pass)")
plt.xlabel("Step Index K")
plt.ylabel("Cos(\u03B8)")
plt.legend([f"Layer {i+1}" for i in range(len(model.layer_sizes))])
plt.grid(True)
plt.savefig(f"{ska_path}/figures/cosine/cosine_history_single_pass.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(np.array(model.frobenius_history).T)  # Z Frobenius
plt.title("Z Tensor Frobenius Norm Evolution Across Layers (Single Pass)")
plt.xlabel("Step Index K")
plt.ylabel("Z Tensor Frobenius Norm")
plt.legend([f"Layer {i+1}" for i in range(len(model.layer_sizes))])
plt.grid(True)
plt.savefig(f"{ska_path}/figures/knowledge/knowledge_frobenius_history_single_pass.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(np.array(model.weight_frobenius_history).T)  # W Frobenius
plt.title("W Tensor Frobenius Norm Evolution Across Layers (Single Pass)")
plt.xlabel("Step Index K")
plt.ylabel("W Tensor Frobenius Norm")
plt.legend([f"Layer {i+1}" for i in range(len(model.layer_sizes))])
plt.grid(True)
plt.savefig(f"{ska_path}/figures/weight/weight_frobenius_history_single_pass.png")
plt.show()

model.visualize_output_distribution()  # Output distribution

model.visualize_net_history()  # Tensor Net historical evolution

print("Training complete. Visualizations generated.")

layers = len(model.layer_sizes)
steps = model.K

save_metric_csv(model.entropy_history, "entropy_history.csv", layers)
save_metric_csv(model.cosine_history, "cosine_history.csv", layers)
save_metric_csv(model.frobenius_history, "frobenius_history.csv", layers)
save_metric_csv(model.weight_frobenius_history, "weight_frobenius_history.csv", layers)
save_metric_csv(model.net_history, "tensor_net_history.csv", layers)

saved_dir = ska_path / "saved"
# Save output history
df_output = pd.DataFrame(
    model.output_history, columns=[f"Class {i}" for i in range(10)]
)
df_output.to_csv(f"{saved_dir}//output_distribution.csv", index_label="Step")
print("Saved output_distribution.csv")
print("All metric data saved. You can now use TikZ in LaTeX to rebuild figures.")
