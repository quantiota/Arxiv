import numpy as np
import torch

from .model import SKAModel
from .wrappers import add_instance_method


@add_instance_method(SKAModel)
def calculate_entropy(model: SKAModel):  # model corresponds to self of the instance
    """Computes entropy reduction, cos(theta), and Tensor Net per layer."""
    total_entropy = 0
    for l in range(len(model.layer_sizes)):  # noqa: E741
        if (
            model.Z[l] is not None
            and model.D_prev[l] is not None
            and model.D[l] is not None
            and model.Z_prev[l] is not None
        ):
            # Compute decision shifts (for entropy)
            model.delta_D[l] = model.D[l] - model.D_prev[l]
            # Compute delta Z (for Tensor Net)
            delta_Z = model.Z[l] - model.Z_prev[l]

            # Compute H_lk as a tensor (element-wise dot product, same shape as D)
            H_lk = (-1 / np.log(2)) * (
                model.Z[l] * model.delta_D[l]
            )  # Element-wise multiplication

            # Compute layer-wise entropy as the sum over all elements
            layer_entropy = torch.sum(H_lk)  # Scalar, for history tracking
            model.entropy[l] = layer_entropy.item()
            model.entropy_history[l].append(layer_entropy.item())

            # Compute cos(theta) for alignment
            dot_product = torch.sum(model.Z[l] * model.delta_D[l])
            z_norm = torch.norm(model.Z[l])
            delta_d_norm = torch.norm(model.delta_D[l])
            if z_norm > 0 and delta_d_norm > 0:
                cos_theta = dot_product / (z_norm * delta_d_norm)
                model.cosine_history[l].append(cos_theta.item())
            else:
                model.cosine_history[l].append(0.0)

            total_entropy += layer_entropy

            # Compute the entropy gradient: nabla_z H = (1/ln2) * z ⊙ D'
            D_prime = model.D[l] * (1 - model.D[l])
            nabla_z_H = (1 / np.log(2)) * model.Z[l] * D_prime

            # Net^(l)_K = delta_Z • (D - nabla_z H)
            tensor_net_step = torch.sum(delta_Z * (model.D[l] - nabla_z_H))
            model.net_history[l].append(tensor_net_step.item())
            model.tensor_net_total[l] += tensor_net_step.item()

    return total_entropy


@add_instance_method(SKAModel)
def ska_update(
    model: SKAModel, inputs, learning_rate=0.01
):  # model corresponds to self of the instance
    """Updates weights using entropy-based learning without backpropagation."""
    for l in range(len(model.layer_sizes)):  # noqa: E741
        if model.delta_D[l] is not None:
            # Previous layer's output
            prev_output = (
                inputs.view(inputs.shape[0], -1) if l == 0 else model.D_prev[l - 1]
            )
            # Compute sigmoid derivative: D * (1 - D)
            d_prime = model.D[l] * (1 - model.D[l])
            # Compute entropy gradient
            gradient = -1 / np.log(2) * (model.Z[l] * d_prime + model.delta_D[l])
            # Compute weight updates via outer product
            dW = torch.matmul(prev_output.t(), gradient) / prev_output.shape[0]
            # Update weights and biases
            model.weights[l] = model.weights[l] - learning_rate * dW
            model.biases[l] = model.biases[l] - learning_rate * gradient.mean(dim=0)
