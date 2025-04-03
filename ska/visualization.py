import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .model import SKAModel
from .wrappers import add_instance_method

path = pathlib.Path().resolve()


# @add_instance_method(SKAModel)
def visualize_entropy_heatmap(model: SKAModel, step):
    """Dynamically scales the heatmap range and visualizes entropy reduction."""
    entropy_data = np.array(model.entropy_history)
    vmin = np.min(entropy_data)  # Dynamically set minimum entropy value
    vmax = 0.0  # Keep 0 as the upper limit for standardization
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        entropy_data,
        cmap="Blues_r",
        vmin=vmin,
        vmax=vmax,
        xticklabels=range(1, entropy_data.shape[1] + 1),
        yticklabels=[f"Layer {i+1}" for i in range(len(model.layer_sizes))],
    )
    plt.title(f"Layer-wise Entropy Heatmap (Step {step})")
    plt.xlabel("Step Index K")
    plt.ylabel("Network Layers")
    plt.tight_layout()
    plt.savefig(f"{path}/figures/entropy/entropy_heatmap_step_{step}.png")
    plt.show()


# @add_instance_method(SKAModel)
def visualize_cosine_heatmap(model: SKAModel, step):
    """Visualizes cos(theta) alignment heatmap with a diverging scale."""
    cosine_data = np.array(model.cosine_history)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cosine_data,
        cmap="coolwarm_r",
        vmin=-1.0,
        vmax=1.0,
        xticklabels=range(1, cosine_data.shape[1] + 1),
        yticklabels=[f"Layer {i+1}" for i in range(len(model.layer_sizes))],
    )
    plt.title(f"Layer-wise Cos(\u03B8) Alignment Heatmap (Step {step})")
    plt.xlabel("Step Index K")
    plt.ylabel("Network Layers")
    plt.tight_layout()
    plt.savefig(f"{path}/figures/cosine/cosine_heatmap_step_{step}.png")
    plt.show()


# @add_instance_method(SKAModel)
def visualize_frobenius_heatmap(model: SKAModel, step):
    """Visualizes the Frobenius Norm heatmap for the knowledge tensor Z across layers."""
    frobenius_data = np.array(model.frobenius_history)
    vmin = np.min(frobenius_data) if frobenius_data.size > 0 else 0
    vmax = np.max(frobenius_data) if frobenius_data.size > 0 else 1
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        frobenius_data,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels=range(1, frobenius_data.shape[1] + 1),
        yticklabels=[f"Layer {i+1}" for i in range(len(model.layer_sizes))],
    )
    plt.title(f"Layer-wise Frobenius Norm Heatmap (Step {step})")
    plt.xlabel("Step Index K")
    plt.ylabel("Network Layers")
    plt.tight_layout()
    plt.savefig(f"{path}/figures/knowledge/knowledge_frobenius_heatmap_step_{step}.png")
    plt.show()


# @add_instance_method(SKAModel)
def visualize_weight_frobenius_heatmap(model: SKAModel, step):
    """Visualizes the Frobenius Norm heatmap for the weight tensors W across layers."""
    weight_data = np.array(model.weight_frobenius_history)
    vmin = np.min(weight_data) if weight_data.size > 0 else 0
    vmax = np.max(weight_data) if weight_data.size > 0 else 1
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        weight_data,
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        xticklabels=range(1, weight_data.shape[1] + 1),
        yticklabels=[f"Layer {i+1}" for i in range(len(model.layer_sizes))],
    )
    plt.title(f"Layer-wise Weight Frobenius Norm Heatmap (Step {step})")
    plt.xlabel("Step Index K")
    plt.ylabel("Network Layers")
    plt.tight_layout()
    plt.savefig(f"{path}/figures/weight/weight_frobenius_heatmap_step_{step}.png")
    plt.show()


# @add_instance_method(SKAModel)
def visualize_output_distribution(model: SKAModel):
    """Plots the evolution of the 10-class output distribution over K steps."""
    output_data = np.array(model.output_history)  # Shape: [K, 10]
    plt.figure(figsize=(10, 6))
    plt.plot(output_data)  # Plot each class as a line
    plt.title("Output Decision Probability Evolution Across Steps (Single Pass)")
    plt.xlabel("Step Index K")
    plt.ylabel("Mean Sigmoid Output")
    plt.legend(
        [f"Class {i}" for i in range(10)], loc="upper right", bbox_to_anchor=(1.15, 1)
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/figures/output/output_distribution_single_pass.png")
    plt.show()


# @add_instance_method(SKAModel)
def visualize_net_heatmap(model: SKAModel, step):
    """Visualizes the per-step Tensor Net heatmap."""
    net_data = np.array(model.net_history)
    vmin = np.min(net_data) if net_data.size > 0 else 0
    vmax = np.max(net_data) if net_data.size > 0 else 1
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        net_data,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        xticklabels=range(1, net_data.shape[1] + 1),
        yticklabels=[f"Layer {i+1}" for i in range(len(model.layer_sizes))],
    )
    plt.title(f"Tensor Net Heatmap (Step {step})")
    plt.xlabel("Step Index K")
    plt.ylabel("Network Layers")
    plt.tight_layout()
    plt.savefig(f"{path}/figures/tensor_net/tensor_net_heatmap_step_{step}.png")
    plt.show()


# @add_instance_method(SKAModel)
def visualize_net_history(model: SKAModel):
    """Plots the historical evolution of Tensor Net across layers."""
    net_data = np.array(model.net_history).T  # Transpose for layer-wise visualization
    plt.figure(figsize=(8, 6))
    plt.plot(net_data)
    plt.title("Tensor Net Evolution Across Layers")
    plt.xlabel("Step Index K")
    plt.ylabel("Tensor Net")
    plt.legend([f"Layer {i+1}" for i in range(len(model.layer_sizes))])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/figures/tensor_net/tensor_net_history_single_pass.png")
    plt.show()


# @add_instance_method(SKAModel)
def visualize_entropy_vs_frobenius(model: SKAModel, step):
    """Plots entropy reduction against Frobenius norm of Z for each layer."""
    plt.figure(figsize=(12, 10))

    # Set up subplots in a 2x2 grid (for 4 layers)
    for l in range(len(model.layer_sizes)):  # noqa: E741
        plt.subplot(2, 2, l + 1)

        # Skip if we don't have enough data points
        if len(model.entropy_history[l]) < 2 or len(model.frobenius_history[l]) < 2:
            plt.title(f"Layer {l+1}: Not enough data")
            continue

        # Get entropy and frobenius data for this layer
        entropy_data = model.entropy_history[l]
        frobenius_data = model.frobenius_history[l][1:]  # Match entropy step indices

        # Ensure same length
        min_len = min(len(entropy_data), len(frobenius_data))
        entropy_data = entropy_data[:min_len]
        frobenius_data = frobenius_data[:min_len]

        # Create scatter plot with connected lines
        plt.scatter(
            frobenius_data,
            entropy_data,
            c=range(len(entropy_data)),
            cmap="Blues_r",
            s=50,
            alpha=0.8,
        )
        plt.plot(frobenius_data, entropy_data, "k-", alpha=0.3)

        # Add colorbar to show step progression
        cbar = plt.colorbar()
        cbar.set_label("Step")

        # Add labels and title
        plt.xlabel("Frobenius Norm of Knowledge Tensor Z")
        plt.ylabel("Entropy Reduction")
        plt.title(f"Layer {l+1}: Entropy vs. Knowledge Magnitude")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{path}/figures/entropy/entropy_vs_frobenius_step_{step}.png")
    plt.show()


global_funs = globals().copy()
visualization_funs = {}

for fun in global_funs.keys():
    if "visualize_" in fun:
        visualization_funs[fun] = global_funs[fun]


class VisualizationManager:
    """Visualization Manager class."""

    def __init__(self, model=SKAModel):
        self.model = model

        for fun_name in visualization_funs.keys():
            setattr(VisualizationManager, fun_name, visualization_funs[fun_name])

        self.wrap_model()

    def wrap_model(self):
        model = self.model
        for fun_name in visualization_funs.keys():
            add_instance_method(model)(visualization_funs[fun_name])


visualizationManager = VisualizationManager()
