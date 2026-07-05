# NECTR: NonExpansive Denoising for ConTractive Reconstruction

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/pdf?id=Z6j8S5LWmL)

Official implementation of **“Trainable Nonexpansive Denoisers for Contractive Image Reconstruction”**, accepted at **International Conference on Machine Learning (ICML 2026)**.

**Authors:**  
[Arghya Sinha](https://arghyasinha.github.io) · 
[Aditya Banerjee](https://in.linkedin.com/in/aditya-banerjee-4253a21b7) · 
[Trishit Mukherjee](https://www.linkedin.com/in/trishit360/) · 
[Kunal N. Chaudhury](https://sites.google.com/site/kunalnchaudhury/home)

NECTR implements a trainable denoiser that is **globally nonexpansive by design** and can be used inside a plug-and-play reconstruction framework to obtain a **contractive reconstruction operator** under mild assumptions.

![NECTR ICML 2026 Poster](https://icml.cc/media/PosterPDFs/ICML%202026/63206.png?t=1783074987.177377)
## Installation

Clone the repository:

```bash
git clone https://github.com/arghyasinha/nectr.git
cd nectr
```

Install the required dependencies:

```bash
pip install torch torchvision numpy scipy scikit-image matplotlib pillow opencv-python hdf5storage einops pyyaml tqdm jupyter
```

For Model 2, which uses a heavier UNet architecture from `deepinv`, also install:

```bash
pip install deepinv
```

## Quick Start

Navigate to the `playgrounds` directory:

```bash
cd playgrounds
```

Open and run the demo notebook:

```bash
jupyter notebook playground.ipynb
```

The notebook provides a simple example of loading a NECTR denoiser and running the reconstruction pipeline.

## Repository Structure

```text
nectr/
├── NECTR_model/
│   ├── NECTR_models1.py
│   ├── NECTR_models2.py
│   ├── model_loader.py
│   ├── configs_model1/
│   │   └── config.yml
│   └── configs_model2/
│       └── config.yml
├── classes/
│   ├── PnP_class.py
│   ├── blur_utils.py
│   ├── utils_image.py
│   └── utils_restoration.py
├── images/
│   ├── CBSD10/
│   └── kernels/
├── playgrounds/
│   └── playground.ipynb
└── README.md
```

## Model Choices for $N_\theta$

The repo currently provide two choices for the neural network $N_\theta$.

### Model 1: Lightweight CNN

Model 1 uses a lightweight CNN architecture.

```text
NECTR_model/configs_model1
```

### Model 2: UNet

Model 2 uses a heavier UNet architecture imported from `deepinv`.

```text
NECTR_model/configs_model2
```

The corresponding configuration files are provided in the above folders. If pretrained checkpoints are not included in this repository, please place the checkpoint files inside the appropriate `checkpoints/` folder before running the demo.

## Choosing a Model in the Playground

In `playground.ipynb`, set the model choice as follows:

```python
MODEL_CHOICE = "model1"  # options: "model1" or "model2"

denoiser_NECTR = load_model(
    config_folder="../NECTR_model/configs_{}".format(MODEL_CHOICE),
    window_rad=7,  # T[7]
    device=device,
    model_choice=MODEL_CHOICE,
).to(device)

run_NECTR = NECTR
```

Here, `window_rad=7` corresponds to the permutation set $T[7]$.

## Using a Custom Architecture

Users can replace $N_\theta$ with their own neural network architecture. To do so, modify the model definition and train the resulting denoiser accordingly. The NECTR framework only requires the neural network to produce the aggregation weights used by the denoiser. Users also can use different permutation sets $\mathcal{G}$. For that, they need to change the `FOR`-loop inside the forward.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{sinha2026trainable,
  title     = {Trainable Nonexpansive Denoisers for Contractive Image Reconstruction},
  author    = {Sinha, Arghya and Banerjee, Aditya and Mukherjee, Trishit and Chaudhury, Kunal Narayan},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```

## Acknowledgements

This work was supported by the Government of India through the PMRF fellowship and ANRF, Qualcomm Technologies, Inc. through the Qualcomm Innovation Fellowship India, and the Kotak IISc AI-ML Centre for GPU resources.
