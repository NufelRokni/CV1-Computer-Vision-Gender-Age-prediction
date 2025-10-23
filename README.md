# Age and Gender Prediction (CNN from Scratch)

Compact Computer Vision project where I built, trained, and evaluated a dual-head CNN that predicts a person’s gender (classification) and age (regression) directly from face images. The work is fully contained in a single, readable notebook and emphasizes clarity, solid baselines, and responsible use.

## One click open

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NufelRokni/CV1-Computer-Vision-Gender-Age-prediction/blob/main/cnn_scratch_age-gender_prediction.ipynb)
[![View in nbviewer](https://img.shields.io/badge/nbviewer-view-blue)](https://nbviewer.org/github/NufelRokni/CV1-Computer-Vision-Gender-Age-prediction/blob/main/cnn_scratch_age-gender_prediction.ipynb)

## Results (validation set in notebook):
- Age MAE: 5.57 years
- Gender Accuracy: 91.2%

## Quickstart

Option A — Run on Colab (recommended)
1. Click the Colab badge above to open the notebook.
2. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU.
3. Mount Google Drive (first cells in the notebook).
4. Point `folder_train_val` to your dataset folder (see Dataset format below).
5. Runtime → Run all.

Option B — Run locally (Windows PowerShell)
```powershell
# 1) Create and activate a virtual environment
# 2) Install dependencies
# 3) Set the dataset path set:
# folder_train_val = "./data/train_val"
# 4) Run all cells (GPU recommended but not required)
```

## Notebook at a glance
Notebook: `cnn_scratch_age-gender_prediction.ipynb`

- Setup and data inspection: Mounts Drive (for Colab) and shows a random grid of labeled faces to sanity-check age and gender labels.
- Data prep: Train/validation split, normalization to [0,1], simple augmentation (horizontal flips).
- Model: Custom CNN with four conv blocks (64→128→256→512) and a shared backbone feeding two outputs:
	- Gender head: sigmoid (binary cross-entropy).
	- Age head: linear (MSE/MAE tracked).
- Training: Adam (lr=0.01), multi-task loss with weights, callbacks (early stopping, multiple ReduceLROnPlateau schedules).
- Evaluation: Reports total loss, task losses, MAE for age, accuracy for gender; includes learning curves.

## Key results
| Task   | Metric | Score   |
|--------|--------|---------|
| Age    | MAE    | 5.57    |
| Gender | Acc.   | 91.2%   |

Figures (you can place them under `assets/` and they’ll render here):
- Learning curves (loss, accuracy, MAE): `assets/learning_curves.png`
- Sample predictions grid: `assets/samples.png`
- Confusion matrix (gender): `assets/confusion_matrix.png`

> If you add these files, they will automatically appear below:

![Learning curves](assets/learning_curves.png)

![Sample predictions](assets/samples.png)

![Confusion matrix](assets/confusion_matrix.png)

## Dataset format
- Expected directory: `train_val/` containing face images.
- Filename pattern encodes labels as: `<age>_<gender>_*.jpg`
	- `age`: integer age (e.g., 23)
	- `gender`: 0 = Male, 1 = Female

In Colab, the default path used is `/content/drive/My Drive/train_val`.
Locally, you can set `folder_train_val = "./data/train_val"` and organize files accordingly.

## Model details
- Input: 128×128×3
- Augmentation: Random horizontal flip
- Backbone: 1×1 conv channel compression → four conv blocks with BatchNorm, ReLU, MaxPool, Dropout
- Head: Dense layers (512 → 256 → 128) with BatchNorm and mixed activations (LeakyReLU, PReLU, SELU)
- Outputs: two heads
	- Gender: Dense(1, sigmoid)
	- Age: Dense(1, linear)

Training setup:
- Optimizer: Adam (lr=0.01)
- Losses: `binary_crossentropy` (gender), `mean_squared_error` (age)
- Loss weights: `{ gender_output: 10, age_output: 1 }`
- Epochs: up to 300, Batch size: 64
- Callbacks: EarlyStopping (patience=50, restore best), ReduceLROnPlateau schedules (training and validation monitored)

## Evaluation and analysis
- Validation metrics (from notebook printout):
	- Total loss ≈ 65.50
	- Gender loss ≈ 0.32, Accuracy ≈ 91.2%
	- Age loss ≈ 62.87, MAE ≈ 5.57
- Visual diagnostics:
	- Learning curves to check convergence and generalization
	- Accuracy curve for gender classification
	- MAE curve for age regression
	- (Optional) Confusion matrix and qualitative error grid

## What I focused on
- Building a strong, readable baseline without heavy transfer learning, to show understanding of CNN building blocks.
- Clear metrics and plots for both tasks to guide iteration.
- Pragmatic regularization and LR scheduling to improve stability.

## Responsible AI and limitations
Predicting age and gender from images involves sensitive attributes and potential bias.

- Intended use: research/learning; not for identity verification or high‑stakes decisions.
- Limitations: Performance may vary across demographics and lighting/pose conditions.
- Mitigations: Balanced sampling, augmentation; evaluate subgroup performance where labels allow.
- Privacy: Ensure consent and compliance with dataset licenses and local laws.

## Project structure
```
.
├─ cnn_scratch_age-gender_prediction.ipynb  # Main notebook (end-to-end)
├─ requirements.txt                         # Minimal dependencies to run locally
├─ assets/                                  # Figures for README
└─ README.md                                # This file
```

## Notes and future improvements
- Extend evaluation with calibration and per-age-bucket analysis.
- Explore transfer learning (e.g., MobileNetV3, EfficientNet) for stronger baselines.

## License
No explicit license is included yet. If you plan to use or extend this work, please open an issue to clarify licensing.

## Contact
You can email me in nufel.rokni.dev@gmail.com for any query. I am happy to discuss enhancement proposals, or any unclear decision in this approach.
