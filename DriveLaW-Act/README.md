# DriveLaW-Act Training and Evaluation

## Data Preparation
Please download the NAVSIM dataset following the instructions in the [installation guide](docs/install.md).

## Stage 1: Diffusion Planner Imitation Learning

You can download our pretrained **DriveLaW-Video** from [DriveLaW-Video](../DriveLaW-Video).  

For the diffusion planner training, the first step is to **cache datasets for faster training**.  
To accelerate, we cache the hidden states output by the Video Model, which enables much faster training.  



### Step 1: Cache hidden states
```bash
# cache dataset for training
sh scripts/evaluation/run_caching_videodrive_hidden_state.sh
```

### Step 2: Configure and run training

Configure the script `scripts/training/run_videodrive_train_stage1.sh` and then start training:

```bash
sh scripts/training/run_videodrive_train_stage1.sh
```

You can also enable **EMA (Exponential Moving Average)** during training for faster convergence in the configuration file. Note that this may lead to very slight performance degradation.


### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh scripts/evaluation/run_videodrive_agent_pdm_score_evaluation_stage1.sh
```

This will evaluate your trained agent using **PDM scores** on the navtest.




## Stage 2: Diffusion Planner Reinforcement Learning Training

In this stage, we perform **reinforcement learning (RL) training** on the Diffusion Planner to further improve planning performance.

### Step 1: Metric Caching

First, you need to cache metrics for the training and test sets, which will be used for evaluation during RL training.


```bash
# cache metrics for navtrain
sh scripts/evaluation/run_metric_caching_train.sh

# cache metrics for navtest
sh scripts/evaluation/run_metric_caching_test.sh
```


### Step 2: Configure and Launch RL Training

After caching metrics, configure the RL training script and launch training:

```bash
# Example path to the RL training script
sh scripts/training/run_videodrive_train_stage2.sh
```

Before running, modify the script parameters as needed according to your hardware and training requirements. This command will start RL training immediately after configuration.


### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh scripts/evaluation/run_videodrive_agent_pdm_score_evaluation_stage2.sh
```
This will evaluate your trained agent using **PDM scores** on the navtest.

