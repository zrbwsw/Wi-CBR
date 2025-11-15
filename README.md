# Wi-CBR: Salient-aware Adaptive WiFi Sensing for Cross-domain Behavior Recognition (AAAI 2026)

This repository provides the official implementation of **Wi-CBR**, a salient-aware adaptive WiFi sensing framework for **cross-domain human behavior / gesture recognition**.  
Wi-CBR jointly exploits **phase** and **Doppler Frequency Shift (DFS)** signals, and uses a **two-branch self-attention** backbone plus a **saliency guidance module** to learn domain-robust representations. 


---

## 1. Datasets

We evaluate Wi-CBR on two large-scale public WiFi sensing datasets:

- **XRF55 Dataset**  
  <https://aiotgroup.github.io/XRF55/>

- **Widar3.0 Dataset**  
  <https://tns.thss.tsinghua.edu.cn/widar3.0/>

Please follow the official dataset licenses and download them from the above pages.

---

## 2. Project Structure

This project contains two main folders:

- `matlab/`  
  - CSI preprocessing and visualization for both **Widar3.0** and **XRF55**  
  - CSI-ratio denoising, STFT to obtain DFS, and generation of 2D images (e.g., `224×224 RGB`) for later deep learning

- `python/`  
  - Our **Wi-CBR** PyTorch implementation  
  - Training / evaluation scripts for cross-domain behavior recognition

> **Important:** in the Python code, please set the **ResNet model** to use ImageNet pretraining, e.g.  
> `pretrained=True` for `ResNet18`.

If you have any questions, feel free to contact: **zrb@mail.hfut.edu.cn**.

---

## 3. Environment

### MATLAB
- R2020a or later (earlier versions should also work if they support STFT and basic plotting)

### Python
- Python ≥ 3.8  
- [PyTorch](https://pytorch.org/) 1.13.1 (with CUDA support if available)  
- Common dependencies:  
  - `numpy`, `scipy`, `matplotlib`, `tqdm`, `scikit-learn` …

You can create a conda environment similar to the one used in the paper:

```bash
conda create -n wi-cbr python=3.9
conda activate wi-cbr
pip install torch==1.13.1 torchvision==0.14.1
pip install numpy scipy matplotlib tqdm scikit-learn
````

---

## 4. Quick Start

### 4.1 Preprocess CSI with MATLAB

1. **Download** Widar3.0 / XRF55 raw CSI data.
2. Place them under the expected directory structure (see comments inside `matlab/` scripts).
3. Run the provided MATLAB scripts to:

   * Apply **CSI-ratio** denoising
   * Extract **phase** and **DFS** (via STFT)
   * Save the results as images (e.g., `224×224 RGB`) or `.mat` feature files

The preprocessed images / features will then be used by the Python code.

### 4.2 Train Wi-CBR with Python

In the `python/` folder:

```bash
python train.py \
  --dataset widar \
  --data_path /path/to/preprocessed_widar \
  --epochs 30 \
  --batch_size 10 \
  --lr 1e-4
```

Key settings (consistent with the paper): 

* Backbone: **ResNet-18**, `pretrained=True` (ImageNet)
* Optimizer: **Adam**, learning rate `1e-4`
* Batch size: `10`
* Epochs: `30`
* Loss: `L_total = L_ce + β · L_con` with `β = 0.1`, temperature `τ = 0.1`
* Same network is used on Widar3.0 and XRF55 for fair comparison.

---

## 5. Widar3.0 File Mapping & Naming Rules

We follow Widar3.0’s naming convention and WIGRUNT’s mapping strategy to locate each sample.

### 5.1 Widar Mapping Method

Raw CSI and DFS files are named as:

* `uname-mn-ln-on-rn-rsn.dat`  (`rsn` ∈ {1, …, 6})
* `envs-suname-mn-ln-on-rn.mat`

where:

* `uname` / `suname` – user ID
* `mn` – gesture ID
* `ln` – location index
* `on` – orientation index
* `rn` – repetition index
* `rsn` – receiver set index

`envs-suname-mn-ln-on-rn.mat` stores a **4D array** in the order:

> **[subcarrier, receiver, transmitter, timestamp]**

### 5.2 Mapping Table (Widar3.0 → WIGRUNT IDs)

For convenience, we adopt the WIGRUNT split:

* **WIGRUNT envs1:** 6750 samples
* **WIGRUNT envs2:** 2249 samples
* **WIGRUNT envs3:** 2997 samples

Some empty files are skipped during loading:

* `12-2-2-3-5`
* `13-1-1-1-1`
* `13-3-3-3-5`
* `14-1-1-1-1`
* `15-3-1-1-5`

#### Detailed Mapping Table

|                                    Dataset File                                    | Env Room |   Action Count   |                     User                    | suname ID | Data Volume |
| :--------------------------------------------------------------------------------: | :------: | :--------------: | :-----------------------------------------: | :-------: | :---------: |
| 20181130_user5_10_11.zip<br>20181130_user12_13_14.zip<br>20181130_user15_16_17.zip |     1    |         9        | User5,10,11<br>User12,13,14<br>User15,16,17 |    0–8    |    10125    |
|                                    20181205.zip                                    |     2    | 2 (gestures 5–6) |                    User2                    |     9     |     250     |
|                                    20181208.zip                                    |     2    | 4 (gestures 1–4) |                    User2                    |     9     |     500     |
|                                    20181205.zip                                    |     2    | 3 (gestures 4–6) |                    User3                    |     10    |     375     |
|                                    20181208.zip                                    |     2    | 3 (gestures 1–3) |                    User3                    |     10    |     375     |
|                                    20181209.zip                                    |     2    |         6        |                    User6                    |     15    |     750     |
|                                    20181204.zip                                    |     2    | 9 (gestures 1–6) |                    User1                    |     16    |     750     |
|                                    20181211.zip                                    |     3    |         6        |                 User3,7,8,9                 |   11–14   |     3000    |

---

## 6. XRF55

We follow exactly the same data protocol as the original XRF55 paper:

> Please refer to the official homepage: [https://aiotgroup.github.io/XRF55/](https://aiotgroup.github.io/XRF55/)

Our cross-domain experiments (cross environment) use scenes 1–4 with quad-fold cross-validation as described in the Wi-CBR paper. 

---

## 7. Feature Set (Precomputed Features)

For convenience, we also release the **feature set** used in the paper:

> [https://drive.google.com/drive/folders/11ANLuUHMFhvqqyXNB1MyXZaTDI_9HbK5?usp=sharing](https://drive.google.com/drive/folders/11ANLuUHMFhvqqyXNB1MyXZaTDI_9HbK5?usp=sharing)

These features contain the preprocessed phase / DFS representations, so you can directly run the Python training code without re-running MATLAB preprocessing.

---

## 8. Citation

If you find this project helpful, please cite:

```bibtex
@inproceedings{zhang2026wicbr,
  title     = {Wi-CBR: Salient-aware Adaptive WiFi Sensing for Cross-domain Behavior Recognition},
  author    = {Ruobei Zhang and Shengeng Tang and Huan Yan and Xiang Zhang and Jiabao Guo},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

---

## 9. Contact

* 💌 Email: **[zrb@mail.hfut.edu.cn](mailto:zrb@mail.hfut.edu.cn)**
* Issues and pull requests are welcome!

