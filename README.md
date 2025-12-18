# CECS 456 — Image Classification with CNNs

## Project Overview

This project builds an **image classification system** using **Convolutional Neural Networks (CNNs)** on the **Natural Images dataset (8 classes)**. The models are implemented in **PyTorch** and compare a **baseline CNN** against a **deeper, regularized CNN**.

The goal is to analyze how **model depth and regularization techniques** impact classification performance and generalization.

---

## Dataset

**Natural Images Dataset (Kaggle)**
[https://www.kaggle.com/datasets/prasunroy/natural-images](https://www.kaggle.com/datasets/prasunroy/natural-images)

* 6,899 RGB images
* 8 classes: airplane, car, cat, dog, flower, fruit, motorbike, person
* Images resized to **160×160** for computational efficiency

The dataset is split into:

* **70%** training
* **15%** validation
* **15%** testing

The train/validation/test indices are saved once (`splits.npz`) and reused by both models to ensure a fair comparison.

---

## Dataset Download & Folder Workflow (Required)

The dataset must be downloaded manually from **Kaggle**.

When downloaded from Kaggle, the file will be named:

```
archive.zip
```

### Step 1: Extract the Kaggle Download

Unzip `archive.zip`. Inside, you will see:

```
archive.zip
├── data/
└── natural_images/
    ├── airplane/
    ├── car/
    ├── cat/
    ├── dog/
    ├── flower/
    ├── fruit/
    ├── motorbike/
    └── person/
```

> **Important:**
> The project expects the folder named **`natural_images/`**.
> The `data/` folder is **not used**.

---

## Models

### Model A — Baseline CNN

* 3 convolutional blocks
* ReLU activations
* Max pooling
* **Global Average Pooling (GAP)** instead of large fully connected layers
* Lightweight and computationally efficient

### Model B — Deeper CNN with Regularization

* 4 convolutional blocks
* Batch Normalization
* Dropout
* Global Average Pooling
* Data augmentation (random crop, flip, rotation)
* Weight decay (L2 regularization)
* Early stopping

---

## Why This Project Runs Efficiently

To ensure the project is **professor-runnable on typical hardware**, the following design choices were made:

* Image size reduced to **160×160**
* Conservative batch size and epoch counts
* `num_workers = 0` to avoid DataLoader issues
* Early stopping to prevent unnecessary training

Typical runtime on a laptop CPU:

* **Model A:** ~30 minutes
* **Model B:** ~8+ hours

---

## CPU vs GPU Runtime Warning

This project **can run on CPU**, but training **will be significantly slower**.

* Running **only on CPU** (e.g., VS Code without GPU) can take **~10× longer**
* Running on **GPU (Google Colab T4)** reduces training from **tens of minutes to a few minutes**

For fastest and most reliable execution, **Google Colab with GPU is strongly recommended**.

---

## Recommended: Google Colab with T4 GPU

Expected runtime on Colab (T4 GPU):

* **Model A:** ~2–5 minutes
* **Model B:** ~5–10 minutes

Using Colab provides:

* GPU acceleration out of the box
* No manual dependency installation
* Faster and more consistent results

---

## Running on Google Colab (Recommended)

### Required Files

* `model_cnn.ipynb`
* `model_deeper.ipynb`
* `splits.npz`
* `natural_images.zip`

---

### Step-by-Step Colab Instructions

1. After extracting `archive.zip` locally, **zip only the `natural_images/` folder**:

   ```
   natural_images.zip
   ```

2. Open **Google Colab** and upload:

   * `model_cnn.ipynb`
   * `model_deeper.ipynb`
   * `natural_images.zip`
   * `splits.npz`

3. Go to **Runtime → Change runtime type** and select **GPU (T4)**.

4. Ensure files are placed as follows:

   * `natural_images.zip` → `/content/sample_data/`
   * `splits.npz` → `/content/splits.npz`

5. Open a Colab terminal and run:

   ```bash
   cd /content/sample_data
   unzip natural_images.zip
   ```

   This creates:

   ```
   /content/sample_data/natural_images/
   ```

6. In both notebooks, set:

   ```python
   DATA_DIR = "/content/sample_data/natural_images"
   ```

7. Run the notebooks **top-to-bottom**.

No additional installation steps are required — **Colab handles all dependencies automatically**.

---

## Running Locally in VS Code (CPU)

### Requirements

If running locally in **VS Code or another IDE**, the following packages **must be installed manually**:

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

> Note: Without GPU support, training will take significantly longer than on Colab.

---

### Local Execution Steps

1. After extracting `archive.zip`, copy the **entire `natural_images/` folder** into your project directory:

   ```
   project_root/
   ├── model_cnn.ipynb
   ├── model_deeper.ipynb
   ├── splits.npz
   └── natural_images/
   ```

2. In both notebooks, set:

   ```python
   DATA_DIR = "Path/to/natural_images"
   SPLIT_FILE = "splits.npz"
   ```

3. Run the notebooks **top-to-bottom**.

---

## Expected Folder Structure Summary

### VS Code

```
project_root/
├── model_cnn.ipynb
├── model_deeper.ipynb
├── splits.npz
└── natural_images/
```

### Google Colab

```
/content/
├── splits.npz
└── sample_data/
    └── natural_images/
```

---

## Key Takeaways

* Deeper architectures with regularization improve generalization.
* Regularization techniques reduce overfitting compared to a baseline CNN.
* GPU acceleration dramatically reduces training time.
* The project remains runnable on CPU for reproducibility and grading.

---

## Author

**Allen Dang**
Solo project — all model design, implementation, experimentation, and analysis were completed independently.
