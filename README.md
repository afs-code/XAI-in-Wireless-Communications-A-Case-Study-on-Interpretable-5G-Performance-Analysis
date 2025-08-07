# XAI-in-Wireless-Communications-A-Case-Study-on-Interpretable-5G-Performance-Analysis

MATLAB implementation of the following paper:

A. F. Şahin, Y. Güven, S. Tedik Başaran, and T. Kumbasar, "XAI in Wireless Communications: A Case Study on Interpretable 5G Performance Analysis," 2025 IEEE 36th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC), Istanbul, Turkey, 2025, pp. 1-6.

We kindly ask that to cite the above mentioned paper if you use methods and publish work papers that was performed using these codes.

## 1)  Requirements

### MATLAB
| Component                                             | Version        |
|-------------------------------------------------------|----------------|
| MATLAB                                                | **R2025a** or newer |
| Deep Learning Toolbox                                 | —              |
| Statistical & Machine Learning Toolbox                | —              |
| Parallel Computing Toolbox                            | —              |

### Python
| Package         | Minimum Version |
|-----------------|-----------------|
| Python          | ≥ 3.7           |
| `numpy`         | —               |
| `pandas`        | —               |
| `scikit-learn`  | —               |
| `interpret`     | Install from <https://github.com/interpretml/interpret> |

---

## 2) Experiments

| Figure (paper) | Script / Notebook | Notes |
|----------------|-------------------|-------|
| **Fig. 1** – Dataset exploration | `sampleInvestigation.m` | — |
| **Fig.2a** –  Ground truth | `sampleInvestigation.m` | — |
| **Fig. 2b** – GAM results        | `gamDemo.m`             | — |
| **Fig. 2c** – EBM results        | `ebmDemo.ipynb`         | Jupyter notebook (Python) |
| **Fig. 2d** – GAMI-Net results   | `gaminetDemo.m`         | **Important:** edit `lib/addInteractions.m` to point to your Python environment and local `interpret` source. |

### Optional: Static-Scenario Experiments
To analyze with static samples, load `Datasets/staticData.mat` or `Datasets/staticData.csv` instead of the default driving dataset.

## 3)  Citation

```bibtex
@inproceedings{sahin2025xai,
  author    = {Ali Fuat Şahin and Yusuf Güven and Semiha Tedik Başaran and Tufan Kumbasar},
  title     = {{XAI} in Wireless Communications: A Case Study on Interpretable 5G Performance Analysis},
  booktitle = {2025 IEEE 36th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)},
  year      = {2025},
  address   = {Istanbul, Turkey},
  pages     = {1--6}
}
