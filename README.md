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

### 📁 Environment Setup

### Default: Using `uv`

Set up the environment using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
uv pip install -e .
```

### Alternative: Using Conda/Mamba 🐍

If you prefer Conda, we recommend using **Mamba** via [Miniforge](https://github.com/conda-forge/miniforge):

#### 1. Create and Activate the Environment

```bash
mamba env create -f conda.yaml
conda activate asymdsd
```

#### 2. Install Module in Editable Mode

```bash
pip install -e .
```

### ⚙️ Additional Notes

#### Installing PyTorch3D

PyTorch3D is built from source (it is not provided as a prebuilt wheel).
If you want CUDA support, you need a CUDA compiler (`nvcc`) available either via a system CUDA Toolkit install or via a Conda-provided `cuda-nvcc`/toolkit package.

If you run into issues during setup, try (exclude `uv` if using Conda/Mamba):

```bash
export MAX_JOBS=4
uv pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
```

> 🐢 Lowering `MAX_JOBS` reduces peak memory usage (slower but useful for systems with limited memory).


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

### ShapeNetPart

Download the ShapeNetPart archive into `data/`:

```bash
cd data
gdown https://drive.google.com/uc?id=1W3SEE-dY1sxvlECcOwWSDYemwHEUbJIS
```

This should create:

- `data/shapenetcore_partanno_segmentation_benchmark_v0_normal.tar`

### ModelNet40 Few-Shot
1. Download the dataset from [ModelNet40 Few-Shot](https://drive.google.com/drive/folders/1gqvidcQsvdxP_3MdUr424Vkyjb_gt7TW?usp=sharing), by selecting all files and downloading them as a zip file `ModelNetFewshot.zip`.
2. Place the zip file in the data folder: `data`.

> ℹ️ Training on the *Mixture* dataset requires additional datasets beyond the ones listed in this section. See [Section 5](#5-dataset-preparation-for-mixture) for the extra preparation steps.


## 3. 🏃‍♂️ Running the Code

### Pre-training

#### AsymDSD-S on ShapeNetCore

To start pretraining the small version of AsymDSD on ShapeNetCore, run:

```bash
sh shell_scripts/sh/train_ssrl.sh
```

> 🧭 You may be prompted to log in to Weights & Biases (wandb).

The first time you run this, it will compile and preprocess the datasets. This process may take a while, but all data is cached under the `data` directory, making subsequent runs much faster.

#### Training with CLS or MPM Modes

To train with specific modes, use the corresponding configuration files:

* **MPM mode**:

  ```bash
  sh shell_scripts/sh/train_ssrl.sh --model configs/ssrl/variants/model/ssrl_model_mask.yaml
  ```

* **CLS mode**:

  ```bash
  sh shell_scripts/sh/train_ssrl.sh --model configs/ssrl/variants/model/ssrl_model_cls.yaml
  ```

#### AsymDSD-B on Mixture

Train the base-sized model on the Mixture dataset:

```bash
sh shell_scripts/sh/train_ssrl.sh --model configs/ssrl/variants/model/ssrl_model_base.yaml --data configs/data/Mixture-U.yaml
```

> 💡 To accelerate pre-training, you can disable evaluation callbacks by editing the trainer config file, or skip all callbacks by passing `--trainer.callbacks null`

### Evaluation
#### Object recognition (ScanObjectNN, ModelNet40)
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

#### Semantic segmentation (ShapeNetPart)

To run semantic segmentation fine-tuning/evaluation, run:

```bash
sh shell_scripts/sh/train_semseg.sh --model.encoder_ckpt_path <ckpt>
```

#### LVIS few-shot (Objaverse-v2)

If you prepared Objaverse-v2 with LVIS annotations (see Section 5), you can run LVIS few-shot evaluation with:

```bash
python shell_scripts/py/Objaverse_fewshot_evals.py --model.encoder_ckpt_path <path_to_model>
```

> Tip: To evaluate a base-sized encoder, add `--model configs/classification/variants/model/classification_model_base.yaml`.


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

## 5. Dataset Preparation for *Mixture*

The *Mixture* dataset is configured in `configs/data/Mixture-U.yaml`.
For most sources, the dataset cache is built automatically the first time you run training (if the raw data is present).

If you want to pre-build caches (or if a dataset is not auto-prepared), use the provided CLI wrapper:

```bash
sh shell_scripts/sh/prepare_data_zarr.sh <dataset_name|config_path>
```

In addition to the datasets listed in Section 2, Mixture uses the following sources.

### Scanned Objects

1. Download the collection using the script shipped in this repository:

```bash
mkdir -p data/ScannedObjects
cd data/ScannedObjects
python ../../asymdsd/data/datasets_/ScannedObjects/download_collection.py \
  -o "GoogleResearch" \
  -c "Scanned Objects by Google Research"
cd ../..
```

2. *Optional*: Prepare the dataset cache:

```bash
sh shell_scripts/sh/prepare_data_zarr.sh ScannedObjects
```

Config: `configs/data/prepare_data_zarr/ScannedObjects.yaml`

### OmniObject3D

1. Go to:
   https://openxlab.org.cn/datasets/OpenXDLab/OmniObject3D-New/tree/main/raw/point_clouds/hdf5_files
2. Create an OpenXLab account and log in if required.
3. Download the `16384` point cloud files.
4. Create `data/OmniObject3D.zip` containing the downloaded files (or rename your downloaded archive to `OmniObject3D.zip`).
5. *Optional*: Prepare the dataset cache (config: `configs/data/prepare_data_zarr/OmniObject3D.yaml`):

```bash
sh shell_scripts/sh/prepare_data_zarr.sh OmniObject3D
```

### 3D-FUTURE

1. Go to:
   https://tianchi.aliyun.com/dataset/98063
2. Create an account, log in, and click **Apply for dataset** to request access.
3. Place the model-part archives in `data/3D-Future/`:
  - `data/3D-Future/3D-FUTURE-model-part1.zip`
  - `data/3D-Future/3D-FUTURE-model-part2.zip`
  - `data/3D-Future/3D-FUTURE-model-part3.zip`
  - `data/3D-Future/3D-FUTURE-model-part4.zip`
4. *Optional*: Prepare the dataset cache (config: `configs/data/prepare_data_zarr/3D-FUTURE.yaml`):

```bash
sh shell_scripts/sh/prepare_data_zarr.sh 3D-FUTURE
```

### Toys4K

1. Go to:
   https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k
2. In the **Downloading Toys4K** section, click the provided link and fill in the form to request access.
3. Place the archive at: `data/toys4k_obj_files.zip`
4. *Optional*: Prepare the dataset cache (config: `configs/data/prepare_data_zarr/Toys4K.yaml`):

```bash
sh shell_scripts/sh/prepare_data_zarr.sh Toys4K
```

### S3DIS Objects

1. Request access by filling in this form:
   https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1
2. Download `Stanford3dDataset_v1.2.zip` and place it at: `data/Stanford3dDataset_v1.2.zip`
3. *Optional*: Prepare the dataset cache (config: `configs/data/prepare_data_zarr/S3DIS_objects.yaml`):

```bash
sh shell_scripts/sh/prepare_data_zarr.sh S3DIS_objects
```

### SUNRGBD

1. Download the required files into `data/SUNRGBD/`:

```bash
mkdir -p data/SUNRGBD
cd data/SUNRGBD
wget https://rgbd.cs.princeton.edu/data/SUNRGBD.zip
wget https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip
wget https://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat
cd ../..
```

2. *Optional*: Prepare the dataset cache (config: `configs/data/prepare_data_zarr/SUNRGBD.yaml`):

```bash
sh shell_scripts/sh/prepare_data_zarr.sh SUNRGBD
```

### Amazon Berkeley Objects (ABO)

1. Download the required files into `data/ABO/`:

```bash
mkdir -p data/ABO
cd data/ABO
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
cd ../..
```

2. *Optional*: Prepare the dataset cache (config: `configs/data/prepare_data_zarr/ABO.yaml`):

```bash
sh shell_scripts/sh/prepare_data_zarr.sh ABO
```

### Objaverse (mandatory, long-running)

Mixture expects a prepared `data/Objaverse.zarr`. Due to the long runtime, this is **not** prepared automatically as part of pre-training.

You can prepare Objaverse in two ways:

1. **Objaverse-v2 (recommended)**: more reliable and faster.

This option also prepares LVIS-based splits/labels used for LVIS few-shot evaluation.

The first time:
```bash
mkdir -p data/ObjaverseV2
sh shell_scripts/sh/prepare_data_zarr.sh Objaverse_v2
```

> Note: No files need to be manually downloaded into `data/ObjaverseV2` for this option. The script will handle downloading and preparing of the raw files.

Config: `configs/data/prepare_data_zarr/Objaverse_v2.yaml`

2. **Objaverse-v1**: use this option if you want to stay closer to replicating the Mixture dataset used in the paper.

```bash
mkdir -p data/Objaverse
sh shell_scripts/sh/prepare_data_zarr.sh Objaverse
```

Config: `configs/data/prepare_data_zarr/Objaverse.yaml`

> Tip: Re-running the command resumes and processes remaining objects.


## 6. 🔮 Future Releases

Planned future releases:

- [x] **Pre-trained Models**: Checkpoints for both small and base versions of **AsymDSD**, including **AsymDSD-CLS** and **AsymDSD-MPM**.
- [x] **Additional Datasets**: Dataset preparation modules including *Mixture* and *Objaverse*.
- [x] **Training Scripts**: Full training configurations for larger model variants and part segmentation on ShapeNet-Part.
