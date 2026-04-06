# 🔥 FeedTTA on VLN-DUET (REVERIE)

This repository extends the original **VLN-DUET** model with **FeedTTA-style online test-time adaptation** for Vision-and-Language Navigation (VLN), evaluated on the **REVERIE** benchmark.

---

## 🚀 Overview

We implement **FeedTTA-style test-time adaptation**, where the agent **adapts online during inference** using **episodic binary feedback** (success / failure).

Unlike standard VLN evaluation (static model), this approach allows the agent to:

- Learn **during navigation in unseen environments**
- Improve performance **without retraining on training data**
- Adapt dynamically to **environment-specific distributions**

---

## 🧠 Method (FeedTTA-style Adaptation)

Our implementation follows the core principles of FeedTTA:

### 1. Episodic Feedback (Core)

After each navigation episode:

- ✅ Success → reward = +1  
- ❌ Failure → reward = -1  

No ground-truth supervision is used during test time.

---

### 2. Online Policy Update

During rollout:

- actions are sampled (`feedback='sample'`)
- log-probabilities are collected

After episode:

- apply **REINFORCE-style update**
- update model parameters online

---

### 3. Stability vs Plasticity

To prevent catastrophic drift during test-time learning:

- Advantage normalization
- Optional entropy regularization
- Optional gradient regularization (SGR-style)

---

## 🔧 Our Extensions

In addition to the base FeedTTA-style setup, we introduce:

### ✅ Progress-Based Reward (Optional)

To improve learning signal:

- Reward = distance reduction toward goal
- + success bonus (navigation + object grounding)

This improves convergence stability on REVERIE.

> ⚠️ Note: This is an extension, not part of original FeedTTA formulation.

---

### ✅ Gradient Regularization (SGR-style, Optional)

We add a lightweight gradient control mechanism:

- Scale gradients differently for:
  - successful episodes
  - failed episodes

Helps balance:

- **plasticity** (adaptation)
- **stability** (retain pretrained knowledge)

---

## 📊 Results (Example)

| Setting        | SR (%) |
|----------------|--------|
| Baseline       | 43.0   |
| + FeedTTA      | 62.3   |

> Results may vary depending on:
> - TTA steps
> - learning rate
> - reward design

---

## 🛠 Setup

### 1. Install Matterport3D Simulator

Follow:
https://github.com/peteanderson80/Matterport3DSimulator

```bash
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH