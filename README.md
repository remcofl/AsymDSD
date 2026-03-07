<!-- markdownlint-disable MD033 MD041 -->
---

<h1 align="center">
AsymDSD
</h1>

<h3 align="center">
Asymmetric Dual Self-Distillation for 3D Self-Supervised Representation Learning
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2506.21724">
    <img src="https://img.shields.io/badge/PDF-arXiv-b31b1b?logo=arxiv&style=flat-square" alt="PDF">
  </a>
  <a href="https://neurips.cc/Conferences/2025">
    <img src="https://img.shields.io/badge/NeurIPS-2025-4c1?style=flat-square" alt="Conference">
  </a>
  <a href="https://github.com/remcofl/AsymDSD">
    <img src="https://img.shields.io/badge/Code-GitHub-blue?style=flat-square&logo=github" alt="Code">
  </a>
  <a href="https://huggingface.co/remcofl/AsymDSD">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow?style=flat-square" alt="HuggingFace">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-97ca00?style=flat-square" alt="License">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square" alt="Python">
</p>

<div align="center">

<table border="0" cellspacing="20">
<tr>
  <td align="center">
    📄 <b>Paper</b><br>
    <a href="https://arxiv.org/abs/2506.21724">arXiv:2506.21724</a>
  </td>

  <td align="center">
    🎓 <b>Conference</b><br>
    NeurIPS 2025 (accepted)
  </td>

  <td align="center">
    🧑🏻‍💼 <b>Authors</b><br>
    Remco F. Leijenaar<br>
    Hamidreza Kasaei
  </td>
</tr>
</table>

</div>

</div>
<p align="center">
  <img src="assets/main_overview.png" width="800" alt="AsymDSD overview"/>
</p>

<p align="center">
  <sub>Overview of the Asymmetric Dual Self-Distillation (AsymDSD) framework.</sub>
</p>

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

Ensure your system provides the following:

* Python 3.11
* GCC/G++ 6–13.3
* C++ build tools (`g++`, `make`; e.g. `build-essential` on Ubuntu)
* Python development headers (`python3-dev`)
* `git`
- CUDA compiler (`nvcc`, from a CUDA Toolkit installation or Conda CUDA package)

> ⚠️ Other versions may work, but only the above configuration has been tested.

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

### 📁 Alternative: Using `uv`

If you prefer not to use Conda, set up the environment using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
uv pip install -e .
```

### ⚙️ Additional Notes

#### Installing PyTorch3D

PyTorch3D is built from source (it is not provided as a prebuilt wheel).
If you want CUDA support, you need a CUDA compiler (`nvcc`) available either via a system CUDA Toolkit install or via a Conda-provided `cuda-nvcc`/toolkit package.

If you run into issues during setup, try:

```bash
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
```

#### Handling Memory Problems During Compilation

If you run out of memory while compiling PyTorch3D, limit parallel build jobs:

For example, this often suffices:

```bash
export MAX_JOBS=4
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
```

> 🐢 Lowering `MAX_JOBS` reduces peak memory usage (slower but more stable).


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

### Larger models
Train 'base'-sized model on the Mixture dataset:
  ```bash
  sh shell_scripts/sh/train_ssrl.sh --model configs/ssrl/variants/model/ssrl_model_base.yaml --data configs/data/Mixture-U.yaml
  ```


> 🔍 You can find logged results on Weights and Biases. A link to the run is provided in the script output. 

## 4. 🤗 Pretrained Checkpoints

We provide AsymDSD pre-trained model checkpoints on Hugging Face:

- Hugging Face repo: https://huggingface.co/remcofl/AsymDSD
- Suggested local folder: `checkpoints/`

| Checkpoint file (HF) | Size | Pre-training data | Variant | Params (M) |
| --- | ---: | --- | --- | --- |
| `AsymDSD-S_ShapeNet.ckpt` | 230 MB | ShapeNetCore | AsymDSD-S | 21.8 |
| `AsymSD-S-CLS_ShapeNet.ckpt` | 197 MB | ShapeNetCore | AsymSD-S-CLS | 21.8 |
| `AsymSD-S-MPM_ShapeNet.ckpt` | 208 MB | ShapeNetCore | AsymSD-S-MPM | 21.8 |
| `AsymDSD-B_Mixture_B.ckpt` | 827 MB | Mixture | AsymDSD-B | 91.8 |

### Download

The environment includes `huggingface_hub`, which provides the `hf` CLI.

Download a single checkpoint into `checkpoints/`:

```bash
hf download remcofl/AsymDSD <model.ckpt> --local-dir checkpoints/
```

If you want to download multiple files, repeat the command with a different filename from the table above.

Once downloaded, you can point evaluation/fine-tuning scripts to the checkpoint, e.g.:

```bash
python shell_scripts/py/train_neural_classifier_all.py --runs <num_eval_runs> --model.encoder_ckpt_path checkpoints/AsymDSD-S_ShapeNet.ckpt
```

## 5. 🔮 Future Releases

Planned future releases:

- [x] **Pre-trained Models**: Checkpoints for both small and base versions of **AsymDSD**, including **AsymDSD-CLS** and **AsymDSD-MPM**.
- [x] **Additional Datasets**: Dataset preparation modules including *Mixture* and *Objaverse*.
- [ ] **Training Scripts**: Full training configurations for larger model variants and part segmentation on ShapeNet-Part.
