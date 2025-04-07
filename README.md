# Structured Knowledge Accumulation (SKA) – ArXiv Repository

## Repository Overview

This repository contains the code and preprint article for the Structured Knowledge Accumulation (SKA) framework, a novel forward-only learning paradigm that eliminates backpropagation and optimizes knowledge accumulation through entropy minimization.

## Issues & Discussions

- Issues: If you encounter problems, report them in the Issues tab.
- Discussions: For theoretical insights, implementation questions, or general feedback, join the Discussions tab.

## Usage

- To run the provided training files, clone the source repository:

```bash
    git clone "https://github.com/quantiota/Arxiv.git"
```

then navigate to the `train.py` file and run it; for Windows:

```bash
    cd "<folder with the cloned repository>/quantiota-Arxiv/scripts"
    python train.py
```

- To use the model outside of the provided `train.py`, add to your imports the following:

```python
    import sys
    import pathlib
    sys.path.append("<path to the folder containing the cloned repo>/quantiota-Arxiv")
    import ska
    from ska.model import SKAModel
    from ska.utils import inputs, save_metric_csv  # inputs containts the provided training data, save_metric_csv saves the results of the training
```

## Contact

For inquiries, reach out at info@quantiota.org.
