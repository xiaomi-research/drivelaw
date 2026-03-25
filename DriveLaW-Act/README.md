# DriveLaW-Act Training and Evaluation

## Stage 1: Data Preparation
Please download the NAVSIM dataset following the instructions in the [installation guide](docs/install.md).

## Stage 2: Diffusion Planner Imitation Learning

You can download our pretrained model from [huggingface](https://huggingface.co/tz2026/DriveLaW) to train DriveLaw-Act by yourself or directly evaluate our pretrained DriveLaW-Act.  

For the diffusion planner training, the first step is to **cache datasets for faster training**. 

To accelerate, we cache the hidden states output by the Video Model, which enables much faster training.  



### Step 1: Cache hidden states and metric
```bash
# cache dataset for training
sh scripts/evaluation/run_caching_videodrive_hidden_state.sh
sh scripts/evaluation/run_metric_caching.sh
```

### Step 2: Configure and run training

Configure the script `scripts/training/run_videodrive_train.sh` and then start training:

```bash
sh scripts/training/run_videodrive_train.sh
```

You can also enable **EMA (Exponential Moving Average)** during training for faster convergence in the configuration file. Note that this may lead to very slight performance degradation.


### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh scripts/evaluation/run_videodrive_agent_pdm_score_evaluation.sh
```

This will evaluate your trained agent using **PDM scores** on the navtest.

> ⚠️ Note: According to the experiment, the performance of DriveLaW-Act is positively correlated with the resolution used during training and inference. All the metrics in the paper were obtained at 1344×768. If your device allows, you can try using a higher resolution to obtain better results. 






