import glob
import os
import pathlib
import shutil
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .model import SKAModel
from .wrappers import add_instance_method

path = pathlib.Path(__file__).parent.parent.resolve()  # This points to the Arxiv directory


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
    plt.title("Output Neuron Activation Evolution Across Steps (Single Pass)")
    plt.xlabel("Step Index K")
    plt.ylabel("Mean Neuron Activation")
    plt.legend(
        [f"Neuron {i}" for i in range(10)], loc="upper right", bbox_to_anchor=(1.17, 1)
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/figures/output/output_neuron_activation_single_pass.png")
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

# @add_instance_method(SKAModel)
def visualize_tensor_net_vs_frobenius(self, step):
    """Plots Tensor Net against Frobenius norm of knowledge tensor Z for each layer."""
    plt.figure(figsize=(12, 10))

    # Set up subplots in a 2x2 grid (for 4 layers)
    for l in range(len(self.layer_sizes)):
        plt.subplot(2, 2, l + 1)

        if len(self.net_history[l]) < 2 or len(self.frobenius_history[l]) < 2:
            plt.title(f"Layer {l + 1}: Insufficient Data")
            continue

        # Tensor Net and Frobenius data
        tensor_net_data = self.net_history[l]
        frobenius_data = self.frobenius_history[l][1:]  # align indices

        # Match lengths
        min_len = min(len(tensor_net_data), len(frobenius_data))
        net_data = self.net_history[l][:min_len]
        frobenius_data = self.frobenius_history[l][1:min_len+1]

        plt.scatter(frobenius_data, net_data, c=range(min_len),
                        cmap='Blues_r', s=50, alpha=0.8)
        plt.plot(frobenius_data, net_data, 'k-', alpha=0.3)

        cbar = plt.colorbar()
        cbar.set_label('Step')

        plt.xlabel('Frobenius Norm of Knowledge Tensor Z')
        plt.ylabel('Tensor Net')
        plt.title(f'Layer {l + 1}: Tensor Net vs. Knowledge Magnitude')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{path}/figures/tensor_net/tensor_net_vs_frobenius_step_{step}.png")
    plt.show()
    

# @add_instance_method(SKAModel)
def visualize_knowledge_flow(self):
        """Plots the Knowledge Flow ||Z_{k+1} - Z_k||_F / learning_rate over steps."""
        plt.figure(figsize=(8, 6))
        for l in range(len(self.layer_sizes)):
            if len(self.knowledge_flow_history[l]) > 0:
                plt.plot(self.knowledge_flow_history[l], label=f"Layer {l+1}")

        plt.title('Knowledge Flow Evolution Across Layers')
        plt.xlabel('Step Index K')
        plt.ylabel('Frobenius Norm of Knowledge Flow Tensor ')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path}/figures/knowledge/knowledge_flow_evolution.png")
        plt.show()


# @add_instance_method(SKAModel)
def visualize_phase_portrait(self):
        """
        Creates a phase portrait by plotting knowledge flow (dZ/dt) against 
        knowledge magnitude (||Z||_F) for each layer, eliminating time as an explicit variable.
        """
        plt.figure(figsize=(12, 10))
        
        # Set up subplots in a 2x2 grid (for 4 layers)
        for l in range(len(self.layer_sizes)):
            plt.subplot(2, 2, l+1)
            
            # Knowledge flow data (dZ/dt)
            flow_data = self.knowledge_flow_history[l]
            
            # Knowledge magnitude data (||Z||_F)
            magnitude_data = self.frobenius_history[l][1:len(flow_data)+1]  # Align indices
            
            if len(flow_data) < 2 or len(magnitude_data) < 2:
                plt.title(f"Layer {l+1}: Insufficient data")
                continue
                
            # Ensure same length
            min_len = min(len(flow_data), len(magnitude_data))
            flow_data = flow_data[:min_len]
            magnitude_data = magnitude_data[:min_len]
            
            # Create scatter plot with connected lines
            plt.scatter(magnitude_data, flow_data, c=range(len(flow_data)), 
                    cmap='Blues_r', s=50, alpha=0.8)
            plt.plot(magnitude_data, flow_data, 'k-', alpha=0.3)
            
            # Add colorbar to show time progression (not as an axis)
            cbar = plt.colorbar()
            cbar.set_label('Time Progression')
            
            # Add labels and title
            plt.xlabel('Frobenius Norm of Knowledge Tensor Z')
            plt.ylabel('Frobenius Norm  of Knowledge Flow')
            plt.title(f'Layer {l+1} Knowledge Flow vs Knowledge Magnitude')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{path}/figures/knowledge/phase_portrait.png")
        plt.show()
        


# @add_instance_method(SKAModel)
def visualize_lagrangian(self):
        """Plots the Lagrangian evolution over steps K for each layer."""
        if not hasattr(self, 'lagrangian_history'):
            print("No Lagrangian history available. Run calculate_lagrangian first.")
            return
        
        plt.figure(figsize=(10, 6))
        for l in range(len(self.layer_sizes)):
            if len(self.lagrangian_history[l]) > 0:
                plt.plot(range(len(self.lagrangian_history[l])), 
                        self.lagrangian_history[l], 
                        label=f"Layer {l+1}")
        
        plt.title('Lagrangian Evolution Across Steps (K)')
        plt.xlabel('Step Index K')
        plt.ylabel('Lagrangian Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path}/figures/lagrangian/lagrangian_evolution.png")
        plt.show()

# @add_instance_method(SKAModel)
def visualize_lagrangian_3d(self):
        """
        Creates only 3D plots of Lagrangian with respect to the Frobenius norms of Z and Phi.
        x-axis: ||Z||_F (Frobenius norm of knowledge tensor) - from frobenius_history
        y-axis: ||Φ||_F (Frobenius norm of knowledge flow tensor) - from knowledge_flow_history
        z-axis: Lagrangian value
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create 3D plots for each layer
        for l in range(len(self.layer_sizes)):
            if (len(self.lagrangian_history[l]) > 0 and 
                len(self.frobenius_history[l]) > 0 and 
                len(self.knowledge_flow_history[l]) > 0):
                
                # The frobenius_history has one more entry than the others
                # because it's calculated before any updates
                # We'll use entries 1:len+1 from frobenius_history to align with others
                
                # Ensure all three lists have the same length
                min_length = min(len(self.lagrangian_history[l]), 
                            len(self.frobenius_history[l]) - 1, 
                            len(self.knowledge_flow_history[l]))
                
                lagrangian_data = self.lagrangian_history[l][:min_length]
                # Use indices 1:min_length+1 from frobenius_history to align with step k
                z_norm_data = self.frobenius_history[l][1:min_length+1]
                phi_norm_data = self.knowledge_flow_history[l][:min_length]
                
                # Create the 3D figure
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create a scatter plot with colors representing progression over steps
                scatter = ax.scatter(z_norm_data, phi_norm_data, lagrangian_data, 
                                c=range(min_length), cmap='viridis', 
                                s=50, alpha=0.7)
                
                # Add a colorbar to show step progression
                cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
                cbar.set_label('Step Index K')
                
                # Try to fit a surface to the data for better visualization
                try:
                    from scipy.interpolate import griddata
                    import numpy as np
                    
                    # Create a grid for interpolation
                    xi = np.linspace(min(z_norm_data), max(z_norm_data), 20)
                    yi = np.linspace(min(phi_norm_data), max(phi_norm_data), 20)
                    xi, yi = np.meshgrid(xi, yi)
                    
                    # Interpolate the Lagrangian values
                    zi = griddata((z_norm_data, phi_norm_data), lagrangian_data, 
                                (xi, yi), method='cubic')
                    
                    # Plot the surface
                    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.5,
                                        linewidth=0, antialiased=True)
                except Exception as e:
                    # Skip surface plot if it fails
                    print(f"Could not fit surface for Layer {l+1} 3D plot: {e}")
                
                # Connect points with a line to show trajectory
                ax.plot(z_norm_data, phi_norm_data, lagrangian_data, 'k-', alpha=0.3)
                
                # Add labels and title
                ax.set_xlabel('Frobenius Norm of Knowledge Tensor')
                ax.set_ylabel('Frobenius Norm of Knowledge Flow')
                ax.set_zlabel('Lagrangian Value')
                plt.title(f'Layer {l+1}: Lagrangian vs Z and Φ Magnitudes')
                
                # Set the viewing angle
                ax.view_init(elev=10, azim=60)
                
                plt.tight_layout()
                plt.savefig(f"{path}/figures/lagrangian/lagrangian_3d_layer_{l+1}.png")
                plt.show()   
    

global_funs = globals().copy()
visualization_funs = {}

for fun in global_funs.keys():
    if "visualize_" in fun:
        visualization_funs[fun] = global_funs[fun]


class VisualizationManager:
    """Visualization Manager class."""

    _path = path

    def __init__(self, model: SKAModel = SKAModel, vispath: Union[str, None] = None):
        self.model = model
        self.path = vispath or self._path
        fun_name: str

        for fun_name in visualization_funs.keys():
            setattr(VisualizationManager, fun_name, visualization_funs[fun_name])

        self.wrap_model()

    def wrap_model(self):
        """Attaches visualization methods to the model."""

        model = self.model
        for fun_name in visualization_funs.keys():
            add_instance_method(model)(visualization_funs[fun_name])

    def delete_visualizations(self, pattern="figures\\*\\*"):
        """Removes png files from the path specified by the pattern."""

        pngfiles = []
        for file in glob.glob(f"{path}\\{pattern}.png"):
            pngfiles.append(file)

        for pngfile in pngfiles:
            try:
                os.remove(pngfile)
            except OSError as e:
                # If it fails, inform the user.
                print(f"Error: {e.filename} - {e.strerror}.")

    def move_visualizations(self, new_path, pattern="figures\\*\\*"):
        """Moves png files from the path specified by the pattern."""

        pngfiles = []
        newpaths = []
        for file in glob.glob(f"{path}\\{pattern}.png"):
            pngfiles.append(file)

            new_file = file[len(path) :]
            new_file = f"{new_path}\\new_file"

            newpaths.append(new_file)

        for pngfile, new_pngfile in zip(pngfiles, newpaths):
            try:
                shutil.move(pngfile, new_pngfile)
            except OSError as e:
                # If it fails, inform the user.
                print(f"Error: {e.filename} - {e.strerror}.")

    def copy_visualizations(self, new_path, pattern="figures\\*\\*"):
        """Copies png files from the path specified by the pattern."""

        pngfiles = []
        newpaths = []
        for file in glob.glob(f"{path}\\{pattern}.png"):
            pngfiles.append(file)

            new_file = file[len(path) :]
            new_file = f"{new_path}\\new_file"

            newpaths.append(new_file)

        for pngfile, new_pngfile in zip(pngfiles, newpaths):
            try:
                shutil.copy(pngfile, new_pngfile)
            except OSError as e:
                # If it fails, inform the user.
                print(f"Error: {e.filename} - {e.strerror}.")


visualizationManager = VisualizationManager()
