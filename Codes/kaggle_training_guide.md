# 🚀 Kaggle Training Guide for PRNet

Training your PRNet models on Kaggle uses free GPU resources (P100 or T4x2).

> **Default**: 100,000 steps (~30 min on P100). For thesis-quality results, use 500,000 steps (~2.5 h).

---

## Step 1: Prepare Your Code

1. Navigate to `d:\ASUFE\Graduation Project\Graduation Project 2\`
2. Right-click the **`Codes`** folder → **Compress to ZIP file**.
3. Name the file **`Codes.zip`**.

---

## Step 2: Upload to Kaggle as a Dataset

1. Log in to [Kaggle](https://www.kaggle.com/).
2. Left sidebar → **Datasets** → **New Dataset**.
3. Enter a title (e.g., `PRNet Source Code`).
4. Drag and drop `Codes.zip` → **Create**. Kaggle auto-unzips it.

---

## Step 3: Create a Kaggle Notebook

1. Go to your new Dataset page → **New Notebook**.
2. Right sidebar → **Session Options → Accelerator**:
   - **GPU P100** — recommended for single model training

> **Note:** No internet access needed. TensorFlow, NumPy, and Matplotlib are pre-installed. No external dependencies.

---

## Step 4: Run the Training

Paste into a new code cell:

```python
import os, shutil

# 1. Clear working directory
for f in os.listdir('/kaggle/working/'):
    p = os.path.join('/kaggle/working/', f)
    os.unlink(p) if os.path.isfile(p) else shutil.rmtree(p)

# 2. Find the Rayleigh folder inside the uploaded dataset
target_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train_prnet.py' in files and 'Rayleigh' in root:
        target_dir = root
        break

# 3. Copy files to the writable working directory
if target_dir:
    for item in os.listdir(target_dir):
        src = os.path.join(target_dir, item)
        dst = os.path.join('/kaggle/working/', item)
        shutil.copytree(src, dst, dirs_exist_ok=True) if os.path.isdir(src) else shutil.copy2(src, dst)
    print(f"✅ Copied from:\n{target_dir}\n")
else:
    print("❌ Could not find the Rayleigh directory.")

# 4. Train — 100k steps (default, ~30 min on P100)
!cd /kaggle/working && python train_prnet.py

# For thesis-quality results, use 500k steps instead (~2.5 h):
# !cd /kaggle/working && python train_prnet.py --steps 500000
```

---

## Step 5: Save & Download Your Trained Models

Once training finishes, weights are saved in `/kaggle/working/models/`.

1. Right sidebar → **Output** → open `models/`.
2. Click **⋯** next to `prnet_lambda_0.01.weights.h5` → **Download**.
3. Place the file into the local `models/` folder of the Rayleigh directory to run `eval_ber.py`, `eval_ccdf.py`, and `plot_constellation.py`.

> **Keep Alive Tip:** Use **"Save Version" → "Save & Run All (Commit)"** to run as a background job. You can close the browser and return later to download — no need to keep the tab open.
