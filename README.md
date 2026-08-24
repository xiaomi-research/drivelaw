<div align="center">

# DriveLaW

**Unifying Planning and Video Generation in a Latent Driving World**

### CVPR 2026

Tianze Xia<sup>1,2*</sup>, Yongkang Li<sup>1,2*</sup>, Lijun Zhou<sup>2*</sup>, Jingfeng Yao<sup>1</sup>, Kaixin Xiong<sup>2</sup>, Haiyang Sun<sup>2†</sup>, Bing Wang<sup>2</sup>,  
Kun Ma<sup>2</sup>, Guang Chen<sup>2</sup>, Hangjun Ye<sup>2</sup>, Wenyu Liu<sup>1</sup>, Xinggang Wang<sup>1✉</sup>

<sup>1</sup> Huazhong University of Science and Technology &nbsp;&nbsp; <sup>2</sup> Xiaomi EV

<sup>*</sup> Equal contribution. &nbsp; <sup>†</sup> Project leader. &nbsp; <sup>✉</sup> Corresponding author.

<br>

<a href="https://arxiv.org/abs/2512.23421"><img src='https://img.shields.io/badge/arXiv-DriveLaW-red' alt='Paper PDF'></a>
<a href="https://xiaomi-research.github.io/drivelaw/"><img src='https://img.shields.io/badge/Project_Page-DriveLaW-green' alt='Project Page'></a>
<a href="https://github.com/xiaomi-research/drivelaw"><img src='https://img.shields.io/badge/Code-GitHub-black' alt='Code'></a>
<a href="https://huggingface.co/tz2026/DriveLaW"><img src='https://img.shields.io/badge/Huggingface-DriveLaW-yellow' alt='Huggingface'></a>

</div>

## News

- **`Aug. 25, 2026`:** Our new work [ReWorld](https://github.com/xiaomi-research/ReWorld) is open-sourced — lower training cost, better results (**FVD 61.9**, **PDMS 90.4**). 🔥
- **`Mar. 25, 2026`:** Weights of DriveLaW-Video and DriveLaW-Act have been released. 🚀
- **`Mar. 25, 2026`:** Codes of DriveLaW have been released. 🚀
- **`Feb. 21, 2026`:** Our paper has been accepted at CVPR 2026. 🎉
- **`Dec. 30, 2025`:** [ArXiv](https://arxiv.org/abs/2512.23421) paper release. Models/Code are coming soon. ☕️

## Updates

- [x] Release Paper
- [x] Release inference & training codes
- [x] Release model weights

## Overview

World models have become crucial for autonomous driving, as they learn how scenarios evolve over time to address the long-tail challenges of the real world. However, current approaches relegate world models to limited roles: they operate within ostensibly unified architectures that still keep world prediction and motion planning as decoupled processes. To bridge this gap, we propose DriveLaW, a novel paradigm that unifies video generation and motion planning. By directly injecting the latent representation from its video generator into the planner, DriveLaW ensures inherent consistency between high-fidelity future generation and reliable trajectory planning. Specifically, DriveLaW consists of two core components: DriveLaW-Video, our powerful world model that generates high-fidelity forecasting with expressive latent representations, and DriveLaW-Act, a diffusion planner that generates consistent and reliable trajectories from the latent of DriveLaW-Video, with both components optimized by a three-stage progressive training strategy. The power of our unified paradigm is demonstrated by new state-of-the-art results across both tasks. DriveLaW not only advances video prediction significantly, surpassing best-performing work by 33.3% in FID and 1.8% in FVD, but also achieves a new record on the NAVSIM planning benchmark.

<div align="center">
  <img src="assets/images/drivelaw-fig2.png" alt="DriveLaW Overview" width="1000">
</div>

## Getting Started

The codebase is organized into two main components:

- **DriveLaW-Video**: Video world model
- **DriveLaW-Act**: Diffusion-based planner that consumes video latents from DriveLaW-Video

Basic installation for DriveLaW with Python 3.10:

```bash
cd DriveLaW-Act
pip install -e .
cd ../DriveLaW-Video/Train
pip install -e .
```

Documentation:

| Component | Guide |
|-----------|-------|
| Video world model (inference) | [DriveLaW-Video/Infer/README.md](DriveLaW-Video/Infer/README.md) |
| Video world model (training) | [DriveLaW-Video/Train/README.md](DriveLaW-Video/Train/README.md) |
| Planning / NavSim evaluation | [DriveLaW-Act/README.md](DriveLaW-Act/README.md) |
| Weights | [Hugging Face — DriveLaW](https://huggingface.co/tz2026/DriveLaW) |

## Contact

If you have any questions, please contact Tianze Xia via email (xiatianze@hust.edu.cn).

## Acknowledgments

DriveLaW is inspired by the following outstanding contributions to the open-source community: [NAVSIM](https://github.com/autonomousvision/navsim), [LTX-Video](https://github.com/Lightricks/LTX-Video), [ReCogDrive](https://github.com/xiaomi-research/recogdrive/tree/main), [Diffusers](https://github.com/huggingface/diffusers), [Genie Envisioner](https://github.com/AgibotTech/Genie-Envisioner/tree/master), [Epona](https://github.com/Kevin-thu/Epona/tree/main).

## Citation

If you find DriveLaW useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@article{xia2025drivelaw,
  title={DriveLaW: Unifying Planning and Video Generation in a Latent Driving World},
  author={Xia, Tianze and Li, Yongkang and Zhou, Lijun and Yao, Jingfeng and Xiong, Kaixin and Sun, Haiyang and Wang, Bing and Ma, Kun and Ye, Hangjun and Liu, Wenyu and others},
  journal={arXiv preprint arXiv:2512.23421},
  year={2025}
}
```
