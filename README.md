# MelodyT5: A Unified Score-to-Score Transformer for Symbolic Music Processing [ISMIR 2024]
This repository contains the code for the MelodyT5 model as described in the paper [MelodyT5: A Unified Score-to-Score Transformer for Symbolic Music Processing](https://arxiv.org/abs/2402.19155).

MelodyT5 is an unified framework for symbolic music processing, using an encoder-decoder architecture to handle multiple melody-centric tasks, such as generation, harmonization, and segmentation, by treating them as score-to-score transformations. Pre-trained on [MelodyHub](https://huggingface.co/datasets/sander-wood/melodyhub), a large dataset of melodies in ABC notation, it demonstrates the effectiveness of multi-task transfer learning in symbolic music processing.

## Model Description
In the domain of symbolic music research, the progress of developing scalable systems has been notably hindered by the scarcity of available training data and the demand for models tailored to specific tasks. To address these issues, we propose MelodyT5, a novel unified framework that leverages an encoder-decoder architecture tailored for symbolic music processing in ABC notation. This framework challenges the conventional task-specific approach, considering various symbolic music tasks as score-to-score transformations. Consequently, it integrates seven melody-centric tasks, from generation to harmonization and segmentation, within a single model. Pre-trained on MelodyHub, a newly curated collection featuring over 261K unique melodies encoded in ABC notation and encompassing more than one million task instances, MelodyT5 demonstrates superior performance in symbolic music processing via multi-task transfer learning. Our findings highlight the efficacy of multi-task transfer learning in symbolic music processing, particularly for data-scarce tasks, challenging the prevailing task-specific paradigms and offering a comprehensive dataset and framework for future explorations in this domain.

We provide the weights of MelodyT5 on [Hugging Face](https://huggingface.co/sander-wood/melodyt5/blob/main/weights.pth), which are based on pre-training with over one million task instances encompassing seven melody-centric tasks. This extensive pre-training allows MelodyT5 to excel in symbolic music processing scenarios, even when data is limited.

## Installation

To set up the MelodyT5 environment and install the necessary dependencies, follow these steps:

1. **Create and Activate Conda Environment**

   ```bash
   conda create --name melodyt5 python=3.7.9
   conda activate melodyt5
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Pytorch**

   ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```
4. **Download Pre-trained MelodyT5 Weights (Optional)**
   
   For those interested in starting with pre-trained models, MelodyT5 weights are available on [Hugging Face](https://huggingface.co/sander-wood/melodyt5/blob/main/weights.pth). This step is optional but recommended for users looking to leverage the model's capabilities without training from scratch.

## Usage

- `config.py`: Configuration settings for training and inference.
- `generate.py`: Perform inference tasks (e.g., generation and conversion) using pre-trained models.
- `train-cls.py`: Training script for classification models.
- `train-gen.py`: Training script for generative models.
- `utils.py`: Utility functions supporting model operations and data processing.
  
   
