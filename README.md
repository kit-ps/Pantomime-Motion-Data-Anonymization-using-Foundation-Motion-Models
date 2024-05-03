# Pantomime: Motion Data Anonymization using Foundation Motion Models (PoPETS 2026)


![Pantomime Example](Pantomime-Examples/Horst-Study/Horst-Study_S43.0011-Gait.npz_vposer_fitting_vposer_0.1.gif).

## Description
This repository contains the code for the paper "Pantomime: Motion Data Anonymization using Foundation Motion Models". Pantomime anonymizes 3D motion data using HuMoR and VPoser motion models. These models require that the motion data be encoded as SMPL parameters. If the motion data is not available in the SMPL format, an additional fitting step is required to convert it. Please note that this process can be time-consuming. In addition to the anonymization code, this repository contains the evaluation code used for identification experiments.

In addition to the code necessary to run Pantomime, this repository contains an overview of motion sequences that have been anonymized using Pantomime, as well as different parameters. The overview can be found [here](Pantomime-Examples/README.md).

Pantomime was implemented reusing the code of [HuMoR](https://geometry.stanford.edu/projects/humor/).

## Environment Setup
> Note: This code was developed on Ubuntu 16.04/18.04 with Python 3.7, CUDA 10.1 and PyTorch 1.6.0. Later versions should work, but have not been tested.

Create and activate a virtual environment to work in, e.g. using Conda:
```
conda create -n humor_env python=3.7
conda activate humor_env
```

Install CUDA and PyTorch 1.6. For CUDA 10.1, this would look like:
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

Install the remaining requirements with pip:
```
pip install -r requirements.txt
```

You must also have _ffmpeg_ installed on your system to save visualizations.

## Downloads & External Dependencies
This codebase relies on various external downloads in order to run for certain modes of operation. Here we briefly overview each and what they are used for. Detailed setup instructions are linked in other READMEs.

### Body Model and Pose Prior 
Detailed instructions to install SMPL+H and VPoser are in [this documentation](./body_models/).

* [SMPL+H](https://mano.is.tue.mpg.de/) is used for the pose/shape body model. Downloading this model is necessary for **all uses** of this codebase.
* [VPoser](https://github.com/nghorbani/human_body_prior) is used as a pose prior only during the initialization phase of fitting, so it's only needed if you are using the test-time optimization functionality of this codebase.

### Datasets
Create for each dataset a folder with the dataset name in the data location (../data/original).

* [Horst-Study](https://data.mendeley.com/datasets/svx74xcrjr/1) gait motion captures. 
* [CeTI-Locomotion](https://springernature.figshare.com/articles/dataset/A_kinematic_dataset_of_locomotion_with_gait_and_sit-to-stand_movements_of_young_adults/26880076?file=48900646) gait and sit-to-stand IMU motion captures.
* [HuMMan](https://huggingface.co/datasets/caizhongang/HuMMan/tree/main/humman_release_v1.0_mogen) An action recognition dataset already in SMPL format. Download the data.zip.

### Pretrained Models
Pretrained model checkpoints are available for HuMoR, HuMoR-Qual, and the initial state Gaussian mixture. To download (~215 MB), from the repo root run `bash get_ckpt.sh`.


## Fitting to Horst-Study and CeTI-Locomotion

Note that this process can take multiple days for both datasets.

To run the fitting of the Horst-Study and CeTI-Locomotion use:
```
python3 humor/fitting/run_fitting_pantomime.py @./configs/fit_ceti_keypts_hyper_opt_full.cfg
python3 humor/fitting/run_fitting_pantomime.py @./configs/fit_horst_keypts_hyper_opt_full.cfg
```

## Prepare SMPL data for anonymization

Before the anonymization the data must be in the right layout to be used for the anonymization dataset.

Run for CeTI-Locomotion and Horst-Study:
```
python3 humor/anonymization/prepare_data.py ../data/  Horst-Study
```

Run for HuMMan:
```
python3 humor/scripts/process_HuMMan.py 
```

## Run anonymization

```
python3 humor/anonymization/run_anonymization.py @./configs/anon_adaptive_HuMMan.cfg
```


