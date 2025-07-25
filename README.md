<div align="center">


<h1 style="display: flex; justify-content: center; align-items: center; gap: 10px; margin: 0;">
LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization
</h1>
<p align="center"><em></em></p>

<p><em>A two-stage RL framework that teaches models to internalize reasoning efficiency.</em></p>

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2507.15758) [![alphaXiv](https://img.shields.io/badge/discussion-A42C25?style=for-the-badge&logo=arxiv&logoColor=white&color=blue
)](https://www.alphaxiv.org/abs/2507.15758) [![Github](https://img.shields.io/badge/LAPO-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/zju-real/LAPO)

</div>

<br>

<div align="center">
  <img src="./figures/overview.png" alt="LAPO Framework" width="85%" />
  <p><em>LAPO's two-stage framework: first discover natural reasoning lengths, then internalize them as self-proposed budgets.</em></p>
</div>

---

## 🎉 News
*   **[2025-7-22]** Our paper, **LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization**, is now available on arXiv!
*   **[Coming Soon]** We plan to release the LAPO-trained models and training configurations. Stay tuned!

---

## Table of Contents
* [Motivation](#motivation)
* [Highlights](#-highlights)
* [Installation](#-installation)
* [Training Pipeline](#-training-pipeline)
* [Results](#-results)
* [Citation](#-citation)
* [Acknowledgement](#-acknowledgement)

---

## Motivation
Large reasoning models often "overthink," generating excessively long and computationally expensive reasoning chains even for simple problems. Existing methods try to fix this with external, rigid constraints, which can harm accuracy and lack adaptability.

<div align="center">
  <img src="./figures/motivation.png" alt="Overthinking Problem" style="width: 90%; height: auto;" />
</div>

LAPO is built on a new paradigm: what if models could learn the appropriate reasoning depth themselves? Our key insight is that the lengths of successful solutions contain valuable signals about a problem's intrinsic complexity. LAPO is designed to:

1.  **Discover** these natural reasoning patterns through a length-aware reward mechanism.
2.  **Internalize** these patterns by framing them as self-proposed plans within the model's own thought process, leading to genuine, adaptive efficiency.

---

## ✨ Highlights

*   💡 **Intrinsic Length Control**: Transforms length control from an external command into an internalized skill. The model learns *when* to think more and *when* to be concise.
*   🚀 **Simultaneous Efficiency & Accuracy Gains**: Reduces token usage by up to **40.9%** while simultaneously improving accuracy by **2.3%** on challenging math benchmarks.
*   ⚙️ **Two-Stage RL Framework**: A robust pipeline that first discovers optimal reasoning lengths (Discovery Stage) and then teaches the model to proactively plan for them (Internalization Stage).
*   📊 **State-of-the-Art Efficiency Frontier**: Outperforms existing methods by achieving a superior balance between accuracy and computational cost.

---

## 🛠 Installation

Our framework is built upon **OpenRLHF**.

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n lapo python=3.10
    conda activate lapo

    # pip install
    pip install openrlhf

    # install vLLM 0.8.5.post1
    pip install vllm==0.8.5.post1
    ```

2.  **Install dependencies:**
    ```bash
    git clone https://github.com/zju-real/LAPO.git
    cd LAPO
    pip install -e .
    pip install -r requirements.txt
    ```
    Please ensure you have a PyTorch version compatible with your CUDA drivers installed.

---

## 🚀 Training Pipeline

LAPO's training is a detailed two-stage process that requires careful configuration. Please follow these steps precisely to replicate our results.

### Step 0: Pre-Training Configuration

Before running any training, you must configure paths in the shell scripts and the reward function file.

**1. Configure Training Scripts:**

Open the following two shell scripts and set the paths for your environment:
*   `./config/deepscaler-1.5B-grpo-stage1.sh`
*   `./config/deepscaler-1.5B-grpo-stage2.sh`

In both files, you **must** modify these variables to match your setup:
*   `pretrain`: Path to the base model.
*   `prompt_data`: Path to your training dataset file.
*   `save_path`: Directory where checkpoints and results will be saved.

**2. Configure Reward Function for Stage 1:**

The reward function needs to know where to save the length statistics collected during Stage 1.

*   **File to edit:** `./examples/scripts/reward_func_stage1.py`
*   **Action:** Inside this file, locate the reward function for Stage 1 and set the path for the output `all_mapping_file` file (json).

### Step 1: Run Stage 1 Training (Discovery)

This stage runs the discovery process to learn the statistical distribution of optimal reasoning lengths.

```bash
bash ./config/deepscaler-1.5B-grpo-stage1.sh
```

Upon completion, this stage will generate a raw `all_mapping_file` file (json) in the output directory you specified.

### Step 2: Process the Length Mapping File

The raw mapping data from Stage 1 must be processed into a clean format for Stage 2. We provide a script, `process.py`, for this task.

**1. Configure Paths in `process.py`:**

Before running the script, you **must** open `process.py` with a text editor and manually update the file path variables at the top. This tells the script where to find your raw data and where to save the final processed file.

**Edit these lines in `process.py`:**

```python
# process.py

# --- Configuration ---
# ‼️ ACTION REQUIRED: Replace these placeholder paths with your actual file paths.

# Input file from Stage 1 training
raw_mapping_file = '/path/to/your/stage1_output/all_mapping_file'

# Final output file for Stage 2 training
final_mapping_file = '/path/to/your/stage1_output/final_mapping_file'

### Step 3: Configure Prompt for Stage 2

For the Internalization stage, the model needs to receive the length guidance in its prompt. You must enable this feature in the dataset code.

*   **File to edit:** `./openrlhf/datasets/prompts_dataset.py`
*   **Action:** In this file, locate the prompt formatting section. **Uncomment the lines** responsible for adding the length-guidance string (e.g., `"I will answer the question with {length} tokens."`) to the prompt.

### Step 4: Run Stage 2 Training (Internalization)

With the final mapping file and the modified prompt logic, you can now start the final training stage. This stage teaches the model to internalize its reasoning budget.

*   **Important:** Before running, ensure the `deepscaler-1.5B-grpo-stage2.sh` script is configured to use the path to your `clean_mapping.json`.

```bash
bash ./config/deepscaler-1.5B-grpo-stage2.sh
```

After this stage, the model in your Stage 2 output directory is the final, LAPO-trained model, capable of adaptive reasoning.

---

## Evaluation

LAPO-trained models consistently achieve higher accuracy with significantly fewer tokens compared to baselines and other efficient reasoning methods. (If you want to evaluate your model, please refer to [lighteval](https://github.com/huggingface/lighteval))

### 📊 Results
The table below presents the Pass@1 accuracy and average token count on MATH-500, AIME2024, AMC-23, and OlympiadBench. LAPO's ability to achieve higher scores with fewer tokens is evident.

| Model / Method | MATH-500 | AIME2024 | AMC-23 | OlympiadBench | **Avg. Pass@1** | **Avg. #Tok** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **_Base: DeepScaleR-1.5B-Preview_** | | | | | | |
| Base Model | 85.8% | 35.5% | 74.2% | 54.6% | 62.5% | 6229 |
| L1-Exact | 80.6% | 24.4% | 70.9% | 48.8% | 56.2% | 2278 |
| L1-Max | 81.9% | 24.9% | 72.7% | 50.5% | 57.5% | 2541 |
| ThinkPrune-4k | 86.6% | 35.5% | 76.3% | 55.7% | 63.5% | 4094 |
| **+ LAPO-D (Ours)** | 86.4% | 37.6% | 77.6% | 56.1% | 64.4% | 4116 |
| **+ LAPO-I (Ours)** | **86.3%** | **38.1%** | **78.3%** | **56.3%** | **64.8%** | **3832** |
| | | | | | | |
| **_Base: DeepSeek-R1-distill-Qwen-1.5B_** | | | | | | |
| Base Model | 83.1% | 30.3% | 68.3% | 50.0% | 57.9% | 8086 |
| **+ LAPO-D (Ours)** | **84.7%** | 28.5% | **72.2%** | 51.3% | **59.2%** | 5177 |
| **+ LAPO-I (Ours)** | 84.3% | 29.3% | 71.2% | **51.7%** | 59.1% | 4775 |

---

## 🙏 Acknowledgement

Our RL training code is built upon the excellent [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) framework. We extend our sincere gratitude to their team for open-sourcing their powerful library.

---
## 📄 Citation

If you find LAPO useful in your research, please consider citing our work:

```bibtex
@misc{wu2025lapointernalizingreasoningefficiency,
      title={LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization}, 
      author={Xingyu Wu and Yuchen Yan and Shangke Lyu and Linjuan Wu and Yiwen Qiu and Yongliang Shen and Weiming Lu and Jian Shao and Jun Xiao and Yueting Zhuang},
      year={2025},
      eprint={2507.15758},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.15758}, 
}
}
```
