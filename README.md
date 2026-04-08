# FeedTTA + Reward Shaping on VLN-DUET (REVERIE)

This project extends **VLN-DUET** with **online test-time adaptation (FeedTTA)** for Vision-and-Language Navigation (VLN), evaluated on the **REVERIE benchmark**.

We further improve adaptation by introducing **continuous reward shaping** and **gradient regularization (SGR)**, leading to significantly improved generalization in unseen environments.

Results and detailed logs are provided in `feedtta-report.ipynb`.

---

## Key Contributions

### 1. FeedTTA-style Online Adaptation
- Performs **policy updates during inference** via REINFORCE  
- Requires **no access to training data at test time**    

---

### 2. Continuous Progress-Based Reward (Main Contribution)

Instead of sparse reward (+1 / -1), we design a **dense reward signal**:

- Reward = **distance reduction toward goal**
- + a small step penalty to discourage inefficient trajectories  
- + a success bonus for correct navigation and grounding  

This design provides:
- more stable learning signal
- better credit assignment across long trajectories  
- improved adaptation performance  

---

### 3. Gradient Regularization (SGR)

Gradient scaling, controlled by `sgr_p` and `sgr_alpha`, were used to stablise test-time adpation and balance plasticity and stability.

---

## Results (REVERIE val_unseen)

### Performance Improvement

| Metric     | Before | After | Δ |
|------------|--------|-------|---|
| SR         | 43.00  | 59.10 | **+16.10** |
| SPL        | 23.07  | 37.42 | +14.34 |
| Oracle SR  | 56.43  | 67.74 | +11.30 |
| RGS        | 28.46  | 39.34 | +10.88 |
| RGSPL      | 15.24  | 24.42 | +9.19 |

---

### Trajectory Efficiency

| Metric  | Before | After | Δ |
|--------|--------|-------|---|
| Steps  | 16.21  | 14.20 | -2.01 |
| Length | 31.36  | 27.37 | -3.99 |

The adapted agent is both **more accurate** and **more efficient**.

---

## Setup


Install Matterport3D Simulator:

https://github.com/peteanderson80/Matterport3DSimulator

## Run

```bash
LD_LIBRARY_PATH=./Matterport3DSimulator/build:$LD_LIBRARY_PATH \
PYTHONPATH=./map_nav_src:./Matterport3DSimulator/build:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
python map_nav_src/reverie/main_nav_obj.py \
  --root_dir ./datasets \
  --dataset reverie \
  --features vitbase \
  --image_feat_size 768 \
  --obj_feat_size 768 \
  --output_dir ./outputs/reverie_feedtta \
  --resume_file ./datasets/REVERIE/trained_models/best_val_unseen \
  --test \
  --batch_size 1 \
  --tta_env_name val_unseen \
  --tta_steps 2 \
  --tta_lr 1e-5 \
  --sgr_p 0.1 \
  --sgr_alpha -0.2 \
  --tta_grad_clip 1.0
  ```

## Key Parameters

| Parameter        | Description |
|------------------|------------|
| tta_steps        | Number of adaptation steps per episode |
| tta_lr           | Test-time learning rate |
| sgr_p            | Probability of applying gradient scaling |
| sgr_alpha        | Gradient scaling factor |
| tta_grad_clip    | Gradient clipping for stability |