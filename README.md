# AsymDSD

## Asymmetric Dual Self-Distillation for 3D Representation Learning
[![ArXiv](https://img.shields.io/badge/arXiv-2506.21724-b31b1b.svg)](https://arxiv.org/abs/2506.21724)
[![Conference](https://img.shields.io/badge/NeurIPS-2025-green.svg)](https://neurips.cc/Conferences/2025)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)]()

- 📄 **Paper**: [Asymmetric Dual Self-Distillation for 3D Self-Supervised Representation Learning](https://arxiv.org/abs/2506.21724)
- 🎓 **Conference**: *NeurIPS 2025* (accepted)
- 🧑🏻‍💼 **Authors**: Remco F. Leijenaar, Hamidreza Kasaei

<p align="center">
  <img src="assets/main_overview.png" alt="AsymDSD Visualization", width="800"/>
</p>  

> **Figure**: Overview of AsymDSD

---

### 📑 Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{leijenaar2025asymmetric,
  title={Asymmetric Dual Self-Distillation for 3D Self-Supervised Representation Learning},
  author={Leijenaar, Remco F and Kasaei, Hamidreza},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 1. 🛠️ Setup Instructions

### 📦 Dependencies

Make sure your system supports the following (some of these will be handled automatically when using the Conda environment, but the system CUDA toolkit is required for building PyTorch3D):

* **Python** 3.11
* **CUDA** 12.4
* **cuDNN** 8.9
* **GCC** version between 6.x and 13.2 (inclusive)

> ⚠️ The code may work with other versions, but only the above configuration has been tested and verified.

### 🐍 Environment Setup

We recommend using **Mamba** via [Miniforge](https://github.com/conda-forge/miniforge) for managing environments.

#### 1. Create and Activate the Environment

```bash
mamba env create -f conda.yaml
conda activate asymdsd
```

#### 2. Install Module in Editable Mode

This allows you to make changes to the source code and see updates without reinstalling.

```bash
pip install -e .
```

### 📁 Alternative: Using `venv` and `pip`

If you prefer not to use Conda, set up the environment using Python's built-in `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation --use-pep517
pip install -e .
```

### ⚙️ Additional Notes

#### Installing PyTorch3D

PyTorch3D requires the CUDA toolkit to be installed and available on your system, even when using Conda. It is not provided as a prebuilt wheel. If you run into issues during setup with Conda, try:

```bash
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation --use-pep517
```

#### Handling Memory Problems During Compilation

If you run out of memory while compiling, remove `ninja`:

```bash
mamba remove ninja
```
and try again.

> 🐢 This will slow down the build process but significantly reduce memory usage.


## 2. 📚 Dataset Preparation
With the provided configurations, dataset files should be placed inside the `data` folder. **Do not extract the archive files**—the code reads directly from the compressed archives.

### ShapeNetCore
1. Request access via [ShapeNet/ShapeNetCore-archive](https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive) on Hugging Face.
2. After approval, download `ShapeNetCore.v2.zip` and place it in the `data` folder.

### ModelNet40
```bash
wget -P data http://modelnet.cs.princeton.edu/ModelNet40.zip
```
> 📖 For more information, visit the [ModelNet40 project page](https://modelnet.cs.princeton.edu/).

### ScanObjectNN

1. Visit the [ScanObjectNN website](https://hkust-vgd.github.io/scanobjectnn/) and agree to the terms of use.
2. Download the dataset `h5_files.zip` and place it in the `data/ScanObjectNN` directory.

```bash
mkdir -p data/ScanObjectNN
# Replace the placeholder below with the actual download link after gaining access
wget -P data/ScanObjectNN <DOWNLOAD_LINK>
```

### ModelNet40 Few-Shot
1. Download the dataset from [ModelNet40 Few-Shot](https://drive.google.com/drive/folders/1gqvidcQsvdxP_3MdUr424Vkyjb_gt7TW?usp=sharing), by selecting all files and downloading them as a zip file `ModelNetFewshot.zip`.
2. Place the zip file in the data folder: `data`.


## 3. 🏃‍♂️ Running the Code

### Pre-training AsymDSD-S on ShapeNetCore

To start pretraining the small version of AsymDSD on ShapeNetCore, run:

```bash
sh shell_scripts/sh/train_ssrl.sh
```

> 🧭 You may be prompted to log in to Weights & Biases (wandb).

The first time you run this, it will compile and preprocess the datasets. This process may take a while, but all data is cached under the `data` directory—making subsequent runs much faster.

### Training with CLS or MPM Modes

To train with specific modes, use the corresponding configuration files:

* **MPM mode**:

  ```bash
  sh shell_scripts/sh/train_ssrl.sh --model configs/ssrl/variants/model/ssrl_model_mask.yaml
  ```

* **CLS mode**:

  ```bash
  sh shell_scripts/sh/train_ssrl.sh --model configs/ssrl/variants/model/ssrl_model_cls.yaml
  ```

> 💡 To accelerate pre-training, you can disable evaluation callbacks by editing the trainer config file, or skip all callbacks by passing `--trainer.callbacks null`

### Evaluation
To evaluate the model on **object recognition tasks**, use the following command:

```bash
python shell_scripts/py/train_neural_classifier_all.py --runs <num_eval_runs> --model.encoder_ckpt_path <path_to_model>
``` 

For the **MPM** pre-trained version without a **CLS**-token, you can add:
```bash
--model configs/classification/variants/model/classification_model_mask.yaml
```

For **few-shot** evaluation on **ModelNet40**:
```bash
python shell_scripts/py/train_neural_classifier_all.py --model.encoder_ckpt_path <path_to_model>
```

> 🔍 You can find logged results on Weights and Biases. A link to the run is provided in the script output. 

## 4. 🔮 Future Releases on Public Repository

We plan on releasing the following resources:

* **Pre-trained Models**: Checkpoints for both small and base versions of **AsymDSD**, including **AsymDSD-CLS** and **AsymDSD-MPM**.
* **Additional Datasets**: Dataset preparation modules including *Mixture* and *Objaverse*.
* **Training Scripts**: Full training configurations for larger model variants and part segmentation on ShapeNet-Part.
