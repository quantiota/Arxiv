import torch
import torch.nn as nn


# Define the SKA model with 4 layers
class SKAModel(nn.Module):
    def __init__(self, input_size=784, layer_sizes=[256, 128, 64, 10], K=50):
        super(SKAModel, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.K = K  # Number of forward steps

        # Initialize weights and biases as nn.ParameterList
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        prev_size = input_size
        for size in layer_sizes:
            self.weights.append(nn.Parameter(torch.randn(prev_size, size) * 0.01))
            self.biases.append(nn.Parameter(torch.zeros(size)))
            prev_size = size

        # Tracking tensors for knowledge accumulation and entropy computation
        self.Z = [None] * len(layer_sizes)  # Knowledge tensors per layer
        self.Z_prev = [None] * len(layer_sizes)  # Previous knowledge tensors
        self.D = [None] * len(layer_sizes)  # Decision probability tensors
        self.D_prev = [None] * len(
            layer_sizes
        )  # Previous decisions for computing shifts
        self.delta_D = [None] * len(layer_sizes)  # Decision shifts per step
        self.entropy = [None] * len(layer_sizes)  # Layer-wise entropy storage

        # Store entropy, cosine, and output distribution history for visualization
        self.entropy_history = [[] for _ in range(len(layer_sizes))]
        self.cosine_history = [[] for _ in range(len(layer_sizes))]
        self.output_history = []  # Store mean output distribution (10 classes) per step

        # Store Frobenius norms for each layer per forward step
        self.frobenius_history = [[] for _ in range(len(layer_sizes))]
        # Store Frobenius norms for each layer's weight matrix W per forward step
        self.weight_frobenius_history = [[] for _ in range(len(layer_sizes))]

        # Store Tensor Net history and total
        self.net_history = [[] for _ in range(len(layer_sizes))]  # Per-step history
        self.tensor_net_total = [0.0] * len(layer_sizes)  # Cumulative total over K

    def forward(self, x):
        """Computes SKA forward pass, storing knowledge and decisions."""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten images

        for l in range(len(self.layer_sizes)):  # noqa: E741
            # Compute knowledge tensor Z = Wx + b
            z = torch.mm(x, self.weights[l]) + self.biases[l]
            # Compute and store Frobenius norm of z
            frobenius_norm = torch.norm(z, p="fro")
            self.frobenius_history[l].append(frobenius_norm.item())
            # Apply sigmoid activation to get decision probabilities
            d = torch.sigmoid(z)
            # Store values for entropy computation
            self.Z[l] = z
            self.D[l] = d
            x = d  # Output becomes input for the next layer
        return x

    def initialize_tensors(self, batch_size):
        """Resets decision tensors at the start of each training iteration."""
        for l in range(len(self.layer_sizes)):  # noqa: E741
            self.Z[l] = None  # Reset knowledge tensors
            self.Z_prev[l] = None  # Reset previous knowledge tensors
            self.D[l] = None  # Reset current decision probabilities
            self.D_prev[l] = None  # Reset previous decision probabilities
            self.delta_D[l] = None  # Reset decision shifts
            self.entropy[l] = None  # Reset entropy storage
            self.entropy_history[l] = []  # Reset entropy history
            self.cosine_history[l] = []  # Reset cosine history
            self.frobenius_history[l] = []  # Reset Frobenius history
            self.weight_frobenius_history[l] = []  # Reset weight Frobenius history
            self.net_history[l] = []  # Reset Tensor Net history
            self.tensor_net_total[l] = 0.0  # Reset Tensor Net total
            self.output_history = []  # Reset output history
