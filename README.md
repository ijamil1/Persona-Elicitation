# Persona Elicitation via Internal Coherence Maximization

This project applies Internal Coherence Maximization (ICM) to the task of **persona elicitation** using the GlobalOpinionQA dataset. The goal is to enable a base language model to search for coherent label assignments in a training set that give it an accurate persona for a given country, then leverage this persona for in-context learning on a held-out test set.

## Acknowledgments

This repository is forked from [Unsupervised-Elicitation](https://github.com/Jiaxin-Wen/Unsupervised-Elicitation) by Jiaxin Wen et al., which implements the ICM algorithm, introduced in https://arxiv.org/pdf/2506.10139v1, for unsupervised elicitation from pretrained language models. The original work demonstrates that ICM is competitive with human-labeled supervision on tasks like TruthfulQA, GSM8k, and Alpaca reward modeling.

## Task: Persona Elicitation with ICM

### Overview

We adapt ICM for **persona elicitation** on a subset of [GlobalOpinionQA](https://huggingface.co/datasets/Anthropic/llm_global_opinions) (GOQA), a dataset containing survey responses from multiple countries on various opinion questions.

**Pipeline:**
1. **Data Preparation**: Extract a subset of questions and countries from GOQA. For each country, split into train and test sets.
2. **ICM Label Search**: For each country, use ICM to search for internally coherent label assignments on the training set. The algorithm iteratively refines labels to maximize mutual predictability within the model's context.
3. **Few-Shot Evaluation**: Use the ICM-discovered labels as demonstrations for many-shot in-context learning on the test set.
4. **Benchmarking**: Compare test accuracy against baselines:
   - Zero-shot (pretrained base model)
   - Zero-shot (chat/instruction-tuned model)
   - Golden supervision (ground truth labels)

### Model

- **Base Model**: Llama 3.1 70B (pretrained, not instruction-tuned)
- **Inference**: Self-hosted via vLLM

### Countries Evaluated

- United States
- France
- Japan

### Key Plots Generated

1. **Test Accuracy Comparison** (per country): Bar chart comparing ICM, golden supervision, and zero-shot baselines
2. **Accuracy vs. Number of Examples**: Line plot showing how test accuracy scales with the number of in-context demonstrations

## Summary of Changes from Original Repository

This fork introduces substantial modifications to adapt ICM for persona elicitation. Key changes:

### New Functionality

- **vLLM Integration** (`core/llm_api/vllm_llm.py`): Added in-process vLLM client for self-hosted inference with FP8 quantization, prefix caching, and batched inference support
- **Data Transformation** (`data_transformation/`): Scripts to transform GlobalOpinionQA into the binary classification format required by ICM:
  - `transform_opinions.py`: Main transformation pipeline
  - `transform_opinions_binary_questions.py`: Variant for binary-only questions
  - `csv_to_json.py`: Format conversion utilities
- **Plotting** (`src/experiments/plot_results.py`): Visualization for test accuracy comparisons and accuracy-vs-examples curves with error bars
- **Test Suite** (`tests/`): Added tests for data loading determinism and cross-country question consistency

### Modified Core Components

- **`src/experiments/ICM.py`**: Major refactoring (+1000 lines changed):
  - Adapted for GOQA persona elicitation task
  - Added country-wise train/test splitting
  - Implemented benchmarking methods (golden supervision, zero-shot baselines)
  - Added accuracy-vs-num-examples comparison functionality
  - Integrated vLLM inference and prefix caching controls
  - Added metrics tracking (acceptance/rejection counts)
  - Multiple iterations for mean/std computation
- **`src/model_querying/prompt_creation.py`**: New prompts tailored for opinion/persona elicitation
- **`core/llm_api/llm.py`**: Refactored to support vLLM alongside OpenAI-compatible APIs
- **`core/llm_api/openai_llm.py`**: Modernized to use AsyncOpenAI client pattern

### Removed Components

- `core/llm_api/anthropic_llm.py`: Removed (not needed for this task)
- `src/runners/query_model.py`: Removed (functionality consolidated)
- `env.yaml`: Replaced with `requirements.txt`

### Parameter Tuning & Optimizations

- Extensive tuning of vLLM parameters (max_model_len, max_num_batched_tokens, max_num_seqs, kv_cache_dtype)
- Temperature scheduling adjustments for simulated annealing
- Prompt engineering iterations for base model compatibility

### Docker Image Build

A separate repository was created to build the custom Docker image: [Docker_PersonaElicitation_Build](https://github.com/ijamil1/Docker_PersonaElicitiation_Build)

**Why this was needed:**
- Installing vLLM via `pip install` was extremely slow on cloud GPU instances, wasting expensive compute credits
- The default `vllm/vllm-openai` images auto-start a vLLM server on container launch, which is incompatible with the in-process vLLM client used in this codebase
- My local machine lacked sufficient disk space to build the image

**Solution:** A minimal Dockerfile that extends the official vLLM image but overrides the entrypoint:

```dockerfile
FROM vllm/vllm-openai:v0.13.0

# Install git and other useful tools
RUN apt-get update && apt-get install -y git curl vim && rm -rf /var/lib/apt/lists/*

# Override the entrypoint so vLLM server doesn't auto-start
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["sleep infinity"]
```

The image is built and pushed to Docker Hub via GitHub Actions, avoiding local disk space constraints.

## Deployment

The experiments were deployed and executed on cloud infrastructure:

- **Platform**: [RunPod](https://www.runpod.io/)
- **Hardware**: 2x NVIDIA B200 GPUs
- **Docker Image**: `irfanjamil10/icm_persona_elicitation` (built on top of `vllm/vllm-openai:v0.13.0`)
- **Inference Framework**: [vLLM](https://github.com/vllm-project/vllm) with prefix caching enabled for efficient batched inference

## Setup

### Hardware Requirements

Running the Llama 3.1 70B model requires significant GPU resources:

- **Model Weights**: ~140 GB (FP16) or ~70 GB (FP8 quantized)
- **KV Cache + Activations**: Additional VRAM required for inference, scaling with context length and batch size
- **Recommended**: Multi-GPU setup with high VRAM (e.g., 2x B200, 4x A100 80GB, or 8x H100)

Be mindful of KV cache and activation memory usage when choosing your GPU configuration. The vLLM parameters `max_model_len`, `max_num_batched_tokens`, and `max_num_seqs` can be tuned to fit within available VRAM.

### Installation

```bash
git clone https://github.com/yourusername/Persona-Elicitation.git
cd Persona-Elicitation
pip install -e .
```

### Secrets

Create a `SECRETS` file at the repository root with your configuration:

```bash
cat > SECRETS << 'EOF'
TOGETHER_API_KEY=<your_together_api_key>
HF_TOKEN=<your_huggingface_token>
NEW_LLAMA_API_BASE=https://api.together.xyz/v1
EOF
```
Note, the Together AI API key is needed for API inference calls to the chat/instruction fine-tuned model to compute the zero-shot chat benchmark.

### Data Preparation

The GlobalOpinionQA data is included in `data/global_opinion_data.csv`. To transform it into the JSON format required by ICM, run the following from the project root:

**Step 1: Transform the CSV data**

There are two transformation scripts depending on which subset of questions you want to use:

- `transform_opinions.py`: Uses all questions (including multi-option questions with 3+ answer choices)
- `transform_opinions_binary_questions.py`: Filters for only binary questions (exactly 2 answer options)

```bash
# Option A: All questions
python3 data_transformation/transform_opinions.py

# Option B: Binary questions only
python3 data_transformation/transform_opinions_binary_questions.py
```

**Step 2: Convert CSV to JSON**

The ICM script reads JSON files, so you must convert the CSV output. The `--dataset` argument must match which transform script you used:

```bash
# If you used transform_opinions.py (all questions):
python3 data_transformation/csv_to_json.py

# If you used transform_opinions_binary_questions.py:
python3 data_transformation/csv_to_json.py --dataset binary
```

This produces JSON files in `data/` with binary labels for each (question, country, option) tuple.

## Running Experiments

### ICM for Persona Elicitation

**Configure model download location (optional but recommended)**

Before running ICM, you may need to specify where vLLM/HuggingFace downloads the model weights. This is important on cloud machines where the default cache location may have insufficient disk space. For example, on RunPod:

```bash
# Create cache directories with sufficient disk space
mkdir -p /workspace/hf_cache
mkdir -p /workspace/hf_cache/hub
mkdir -p /workspace/hf_cache/transformers
mkdir -p /workspace/tmp

# Set environment variables
export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export TMPDIR=/workspace/tmp
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on your GPU setup
```

**Run ICM**

```bash
cd src/experiments

# 8B model with 1 GPU
python ICM.py \
    --model meta-llama/Llama-3.1-8B \
    --tensor_parallel_size 1 \
    --kv_cache_dtype auto \
    --gpu_memory_utilization 0.90 \
    --enable_chunked_prefill True \
    --max_num_batched_tokens 131072 \
    --consistency_fix_K 10 \
    --K 1000

# 8B FP8 quantized model with 2 GPUs
python ICM.py \
    --model RedHatAI/Meta-Llama-3.1-8B-FP8 \
    --tensor_parallel_size 2 \
    --kv_cache_dtype auto \
    --gpu_memory_utilization 0.90 \
    --enable_chunked_prefill True \
    --max_num_batched_tokens 131072 \
    --consistency_fix_K 10 \
    --K 1000

# 70B FP8 quantized model with 4 GPUs
python ICM.py \
    --model RedHatAI/Meta-Llama-3.1-70B-FP8 \
    --tensor_parallel_size 4 \
    --kv_cache_dtype auto \
    --max_num_batched_tokens 131072 \
    --enable_chunked_prefill True \
    --gpu_memory_utilization 0.95 \
    --consistency_fix_K 10 \
    --K 1000

# 70B model with 4 GPUs
python3 ICM.py \
    --model meta-llama/Llama-3.1-70B \
    --tensor_parallel_size 4 \
    --enable_chunked_prefill True \
    --gpu_memory_utilization 0.85 \
    --consistency_fix_K 20 \
    --K 1000

# 70B model with 2 GPUs
python3 ICM.py \
    --model meta-llama/Llama-3.1-70B \
    --tensor_parallel_size 2 \
    --kv_cache_dtype auto \
    --enable_chunked_prefill True \
    --gpu_memory_utilization 0.85 \
    --consistency_fix_K 20 \
    --K 1500

# 70B model with 2 GPUs and binary data
python3 ICM.py \
    --model meta-llama/Llama-3.1-70B \
    --tensor_parallel_size 2 \
    --kv_cache_dtype auto \
    --enable_chunked_prefill True \
    --gpu_memory_utilization 0.85 \
    --consistency_fix_K 20 \
    --K 1000 \
    --dataset binary
```

**ICM Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `meta-llama/Llama-3.1-70B` | HuggingFace model name or path |
| `--alpha` | `3.5` | Coefficient for mutual predictability in ICM scoring function |
| `--num_seed` | `12` | Number of randomly labeled examples to initialize ICM |
| `--K` | `1000` | Maximum ICM iterations |
| `--consistency_fix_K` | `20` | Maximum iterations for consistency fix phase |
| `--decay` | `0.995` | Decay rate for simulated annealing |
| `--initial_T` | `10` | Initial temperature for simulated annealing |
| `--final_T` | `0.1` | Final temperature for simulated annealing |
| `--scheduler` | `log` | Decay scheduler for simulated annealing |
| `--dataset` | `all` | Dataset to use: `all` for all questions, `binary` for binary questions only |

**vLLM Configuration:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `--kv_cache_dtype` | `fp8` | Data type for KV cache (e.g., `fp8`, `auto`) |
| `--enable_chunked_prefill` | `False` | Enable chunked prefill for better memory efficiency |
| `--gpu_memory_utilization` | `0.90` | Fraction of GPU memory to use (0.0-1.0) |
| `--max_model_len` | `15000` | Maximum number of tokens that a sequence can grow to |
| `--max_num_batched_tokens` | `115072` | Maximum number of batched tokens per forward pass of the model |
| `--max_num_seqs` | `64` | Maximum number of active sequences |

### Generating Plots

After running experiments, `ICM.py` prints JSON-friendly output to stdout. To generate plots:

1. Save the JSON output to a file in `src/results/`:
```bash
cd ..
mkdir -p src/results
# Copy the JSON output from ICM.py stdout and save it to a file, e.g.:
# src/results/my_results.json
```

2. Run the plotting script from the results directory:
```bash
cd src/results
python3 ../experiments/plot_results.py --input my_results.json
```

This generates `.png` plot files in `src/results/`.

## File Structure

```
.
├── core/
│   └── llm_api/
│       ├── llm.py              # Model API abstraction
│       ├── vllm_llm.py         # vLLM in-process client (NEW)
│       └── openai_llm.py       # OpenAI-compatible API client
├── data/
│   ├── global_opinion_data.csv # Source GOQA data
│   └── transformed_*.csv/json  # Processed data
├── data_transformation/        # Data processing scripts (NEW)
├── src/
│   ├── experiments/
│   │   ├── ICM.py              # Main experiment script
│   │   ├── ICM_tools.py        # ICM helper functions
│   │   └── plot_results.py     # Visualization (NEW)
│   ├── model_querying/
│   │   ├── prompt_creation.py  # Prompt templates
│   │   └── solution_extraction.py
│   ├── pipeline/
│   │   └── pipeline.py         # Inference pipeline
│   └── results/                # Output JSON and plots (NEW)
└── tests/                      # Test suite (NEW)
```

## Citation

If you use this work, in addition to acknowledging this repo, please mention the original ICM paper and repo:

https://arxiv.org/pdf/2506.10139v1

https://github.com/Jiaxin-Wen/Unsupervised-Elicitation

