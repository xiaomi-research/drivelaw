<div align="center">

# DriveLaW

**Unifying Planning and Video Generation in a Latent Driving World**

<span class="venue-pill">CVPR 2026</span>

Tianze Xia<sup>1,2*</sup>, Yongkang Li<sup>1,2*</sup>, Lijun Zhou<sup>2*</sup>, Jingfeng Yao<sup>1</sup>, Kaixin Xiong<sup>2</sup>, Haiyang Sun<sup>2†</sup>, Bing Wang<sup>2</sup>,  
Kun Ma<sup>2</sup>, Guang Chen<sup>2</sup>, Hangjun Ye<sup>2</sup>, Wenyu Liu<sup>1</sup>, Xinggang Wang<sup>1✉</sup>

<sup>1</sup> Huazhong University of Science and Technology &nbsp;&nbsp; <sup>2</sup> Xiaomi EV

<sup>*</sup> Equal contribution. &nbsp; <sup>†</sup> Project leader. &nbsp; <sup>✉</sup> Corresponding author.

<br>

<a class="link-btn" href="https://arxiv.org/abs/2512.23421">📄 Paper</a>
<a class="link-btn" href="https://xiaomi-research.github.io/drivelaw/">🌐 Project Page</a>
<a class="link-btn" href="https://github.com/xiaomi-research/drivelaw">💻 Code</a>
<a class="link-btn" href="https://huggingface.co/tz2026/DriveLaW">🤗 Weights</a>

</div>

<style>
.venue-pill {
  display: inline-block;
  padding: 6px 16px;
  border: 1px solid #dbe3ee;
  border-radius: 999px;
  background: #ffffff;
  color: #475569;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.link-btn {
  display: inline-block;
  margin: 4px 6px;
  padding: 8px 18px;
  border: 1px solid #dbe3ee;
  border-radius: 10px;
  background: #ffffff;
  color: #111827;
  font-size: 14px;
  font-weight: 600;
  text-decoration: none;
  transition: all .18s ease;
}
.link-btn:hover {
  border-color: #94a3b8;
  box-shadow: 0 8px 20px rgba(15, 23, 42, .08);
  transform: translateY(-1px);
  text-decoration: none;
}
</style>

## News

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
