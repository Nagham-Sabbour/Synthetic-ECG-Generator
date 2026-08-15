# Synthetic-ECG-Generator
Implementation of a deep generative model using a conditional VAE and GAN-based finetuning to generate realistic, class-specific, synthetic ECG signals.

## Setup
Clone this repo to your preferred location.

### Installation
Install the requirements listed in requirements.txt using the following:

`pip install -r requirements.txt`

### Data
Data has been preprocessed using `notebooks/PTBXL_Preprocessing.ipynb` and stored in `/processed_data`. 

If the raw PTB-XL data is needed for any separate processing, it can be downloaded from the following link:

https://physionet.org/content/ptb-xl/1.0.3/

## Usage
### VAE Training
To train the base VAE model, run the following command `python train.py` with the following optional arguments:
- `--data-root` (Path to folder with processed data)
- `--batch-size` (Batch size, should be divisible by the number of classes 11)
- `--epochs` (Number of epochs to train for)
-  `--lr` (Learning rate)
- `--embedding-dim` (Embedding/Latent dimension)
- `--loss-beta` (Weight for the KL term in the loss)
- `--num-classes` (Number of classes)
- `--checkpoint-dir` (Directory to save the model checkpoints)
- `--visuals-dir` (Directory to save the visuals for generated and reconstructed samples)
- `--plots-dir` (Directory to save the loss plots)

#### Outputs
Default locations of training outputs are as follows:
- Model checkpoints (best and final) can be found in `/checkpoints`
- Visuals of generated and reconstructed samples can be found in `/visuals`
- Loss curve plots can be found in `/training_plots`

### GAN Finetuning
To run the GAN finetuning, run the following command `python finetune.py` with the following arguments:
- `--trained-vae-filename` (Required. Filename of trained VAE checkpoint)
- `--data-root` (Path to folder with processed data)
- `--batch-size` (Batch size, should be divisible by the number of classes 11)
- `--epochs` (Number of epochs to train for)
- `--decoder-lr` (Learning rate for the decoder)
- `--discrim-lr` (Learning rate for the discriminator)
- `--embedding-dim` (Embedding/Latent dimension, must match the original VAE training run)
- `--loss-lambda-adv` (Weight for the adversarial term in the generator loss)
- `--num-classes` (Number of classes, must match the original VAE training run) 
- `--checkpoint-dir` (Directory to save the model checkpoints)
- `--visuals-dir` (Directory to save the visuals for generated and reconstructed samples)
- `--plots-dir` (Directory to save the loss plots)

#### Outputs
Default locations of finetuning outputs are as follows:
- Model checkpoints (best and final) can be found in `/checkpoints`
- Visuals of generated and reconstructed samples can be found in `/visuals`
- Loss curve plots can be found in `/training_plots`

### Evaluation metrics
To run the computation of the evaluation metrics, run the following command `python evaluate.py` with the following arguments:
- `--data-root` (Path to folder with processed data)
-  `--checkpoint-path` (Required. Path to the model checkpoint on which to perform evaluation)  
- `--results-root` (Directory to save the evaluation results)
- `--embedding-dim` (Embedding/Latent dimension, must match the model)
- `--samples-per-class` (Number of samples to use for each class)
- `--seed` (Seed for reproducible runs)

#### Outputs
Default location of the evaluation results is `/test_runs`.

### TSTR
To run the TSTR (train on synthetic, test on real), run the following command `python tstr.py` with the following arguments:
- `--data-root` (Path to folder with processed data)
-  `--checkpoint-path` (Required. Path to the model checkpoint on which to perform tstr) 
- `--results-root` (Directory to save the tstr results)
- `--embedding-dim` (Embedding/Latent dimension, must match the model)
- `--real-samples-per-class` (Number of real samples to use per class)
- `--synthetic-samples-per-class` (Number of synthetic samples to use per class)
- `--mixed-synthetic-samples-per-class` (Number of synthetic samples to use per class combined with 300 real)
- `--seed` (Seed for reproducible runs)
- `--classifier-epochs` (Number of epochs to run the classifier)
- `--classifier-patience` (Number of epochs to wait for classifer improvement before early stopping)
- `--batch-size` (Batch size)
- `--learning-rate` (Learning rate)

#### Outputs
Default location of the TSTR results is `/test_runs`.

## Demo Notebook
A demo notebook (`/demo/demo_notebook.ipynb`) for showing sample input-output was run. It was run in Google Colab with T4 GPU includes the following flow:
- Environment setup and installations
- Train the VAE
- Run evaluation metrics on trained VAE
- Run TSTR on trained VAE
- Finetune the VAE using GAN
- Run evaluation metrics on finetuned model
- Run TSTR on finetuned model

The outputs generated from these steps got saved to subfolders within `/demo`, such as `/demo/checkpoints`, `/demo/training_plots`, `/demo/visuals` and `/demo/test_runs`. 