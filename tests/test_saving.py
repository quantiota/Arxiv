import glob
import os
import pathlib
from os.path import isfile

import matplotlib

from ska.model import SKAModel

model = SKAModel()
path = pathlib.Path().resolve()
matplotlib.use("Agg")

pattern = "figures\\*\\*_test*"
pngfiles = []
for file in glob.glob(f"{path}\\{pattern}.png"):
    pngfiles.append(file)

for pngfile in pngfiles:
    try:
        os.remove(pngfile)
    except OSError as e:
        # If it fails, inform the user.
        print(f"Error: {e.filename} - {e.strerror}.")


def test_save_entropy_heatmap():
    steps = ["test1", "test2"]

    model.layer_sizes = [1, 1, 1, 1]
    model.entropy_history = [[0] for _ in range(len(model.layer_sizes))]

    for step in steps:
        model.visualize_entropy_heatmap(step)
        assert isfile(f"{path}/figures/entropy/entropy_heatmap_step_{step}.png") is True


def test_save_cosine_heatmap():
    steps = ["test1", "test2"]

    model.layer_sizes = [1, 1, 1, 1]
    model.cosine_history = [[0] for _ in range(len(model.layer_sizes))]

    for step in steps:
        model.visualize_cosine_heatmap(step)
        assert isfile(f"{path}/figures/cosine/cosine_heatmap_step_{step}.png") is True

    pass


def test_save_frobenius_heatmap():
    steps = ["test1", "test2"]

    model.layer_sizes = [1, 1, 1, 1]
    model.frobenius_history = [[0] for _ in range(len(model.layer_sizes))]

    for step in steps:
        model.visualize_frobenius_heatmap(step)
        assert (
            isfile(
                f"{path}/figures/knowledge/knowledge_frobenius_heatmap_step_{step}.png"
            )
            is True
        )

    pass


def test_save_weight():
    steps = ["test1", "test2"]

    model.layer_sizes = [1, 1, 1, 1]
    model.weight_frobenius_history = [[0] for _ in range(len(model.layer_sizes))]

    for step in steps:
        model.visualize_weight_frobenius_heatmap(step)
        assert (
            isfile(f"{path}/figures/weight/weight_frobenius_heatmap_step_{step}.png")
            is True
        )
    pass


def test_save_output():
    model.K = 1
    model.output_history = [index for index in range(10)]

    model.visualize_output_distribution()
    assert isfile(f"{path}/figures/output/output_distribution_single_pass.png") is True
    pass


def test_save_net_heatmap():
    steps = ["test1", "test2"]

    model.layer_sizes = [1, 1, 1, 1]
    model.net_history = [[0] for _ in range(len(model.layer_sizes))]

    for step in steps:
        model.visualize_net_heatmap(step)
        assert (
            isfile(f"{path}/figures/tensor_net/tensor_net_heatmap_step_{step}.png")
            is True
        )
    pass


def test_save_net_history():
    model.layer_sizes = [1, 1, 1, 1]
    model.net_history = [[0] for _ in range(len(model.layer_sizes))]

    model.visualize_net_history()
    assert (
        isfile(f"{path}/figures/tensor_net/tensor_net_history_single_pass.png") is True
    )

    pass


def test_save_entropy_vs_frob():
    steps = ["test1", "test2"]

    model.layer_sizes = [1, 1, 1, 1]
    model.frobenius_history = [[0] for _ in range(len(model.layer_sizes))]
    model.entropy_history = [[0] for _ in range(len(model.layer_sizes))]

    for step in steps:
        model.visualize_entropy_vs_frobenius(step)
        assert (
            isfile(f"{path}/figures/entropy/entropy_vs_frobenius_step_{step}.png")
            is True
        )

    pass
