# Transition Matching on Anime

[Transition Matching Paper](https://arxiv.org/abs/2506.23589v1)

## Guidelines

1. Favor simple functions over complex abstractions.
2. Prefer standalone functions; minimize class usage.
3. Use Git branches to separate and develop individual tasks.

---

## Prerequisites

- **Task 1**: Read the TM Paper - DONE
- **Task 2:** Review this [repo](https://github.com/gstoica27/DeltaFM) for best practices - DONE

  - How they perform the pre-processing
  - How they are sampling from $p_t$

---

## Project Tasks

### Task 1: Setup Environment

1. **Project Structure**

   ```text
   flow_anime/
   ├── notebook/
   ├── ckpt/
   ├── data/
   └── src/
       ├── __main__.py
       ├── __init__.py
       ├── workflow.py
       ├── sample.py
       ├── model.py
       ├── train.py
       ├── eval.py
       └── app.py
   ```

2. **CLI Commands** (using `cyclopts`)

   - `train` — Start model training (with configurable options)
   - `generate` — Generate output from a prompt via ODE solver
   - `eval` — Evaluate model performance against metrics
   - `logging` — Launch local logging server
   - `workflow` — Launch node-based web UI

3. **Configuration**

   - Merge YAML config with CLI options.
   - Toggle debugging (verbose logs).
   - Swap or customize model and autoencoder.
   - Change text encoder (e.g., CLIP, T5Gemma).
   - Configure training parameters (learning_rate, batch_size, schedule).
   - Enable save/resume training checkpoints.
   - Set random seed and compute device.
   - Select loss functions (single or composite).
   - Add advanced techniques (e.g., DDT).
   - Specify dataset for text-to-image training.

4. **Logging & Monitoring**

   - Integrate with Weights & Biases (wandb).
   - Log training losses and gradient statistics (pre‑/post‑activation, norms).
   - Evaluate model using standard metrics.
   - Generate sample images with fixed and random noise via low‑step ODE.

---

### Task 2: Analyze Anime Dataset

1. Create dataset class
2. Download dataset locally
3. Review dataset text and images
4. Perform exploratory data analysis
5. Review literature on how to handle various aspect ratios

---

### Task 3: Integrate External Models

#### Subtask 3.1: Image Latent Encoder (DC-AE | DeTok)

1. Design architecture.
2. Define encoding pipeline.
3. Adapt data preprocessing and transformations.

#### Subtask 3.2: Text Encoder - T5Gemma

1. Summarize key paper contributions.
2. Define tokenization and embedding process.
3. Integrate into diffusion pipeline (reference T5 usage).
4. Evaluate on anime-tag dataset.

#### Subtask 3.3: Text Encoder - CLIP

1. Research latest CLIP variants.
2. Define encoding workflow.

---

### Task 4: Implement DTM Architecture

- Translate DTM design into code modules.
- Ensure modularity for ease of testing.

---

### Task 5: Training Pipeline

1. **Core Training Loop**

   1. Review literature on multi-scale diffusion models
   2. Implement aspect ratio bucketing?

2. **Exponential Moving Average (EMA)**
3. **Distributed Training**

   - Options: DDP, FSDP, NCCL, DeepSpeed
   - References:
     1. PyTorch Distributed Training (YouTube)
     2. PyTorch GitHub Issue #114299
     3. FSDP Tutorial (PyTorch Docs)
     4. FSDP API Reference

---

### Task 6: Convergence Techniques

#### Subtask 6.1: Decoupled Denoising Training (DDT)

- Reference: [https://arxiv.org/abs/2504.05741](https://arxiv.org/abs/2504.05741)

#### Subtask 6.2: Dispersion Loss

- Reference: [https://arxiv.org/abs/2506.09027](https://arxiv.org/abs/2506.09027)

#### Subtask 6.3: Contrastive Loss

- Reference: [https://arxiv.org/abs/2506.05350](https://arxiv.org/abs/2506.05350)

#### Subtask 6.4: Mean Flows

- Reference: [https://arxiv.org/abs/2505.13447](https://arxiv.org/abs/2505.13447)

---

### Task 7: LoRA Fine-Tuning

1. Implement LoRA variants

   1. [LoRA](https://arxiv.org/abs/2106.09685)
   2. [T-LoRA](https://arxiv.org/abs/2507.05964)

2. Subject-specific fine-tuning.
3. Style-specific fine-tuning.

---

### Task 8: Post-Training using DPO

1. Explore DPO techniques
2. Implement two DPO variants

---

### Task 9: Model Quantization using torchao

1. Perform quantization

---

### Task 10: Node-Based Workflow App

1. **Backend**: FastAPI
2. **Frontend**: Astro
3. **Node UI**: React Flow
4. **Node Logic Components**:

   - ODE Simulation Viewer
   - Model Loader
   - Text Prompt Input
   - Sampler Controller
