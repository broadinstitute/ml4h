This folder contains the code and notebooks used in our paper: ["Genetic Architectures of Medical Images Revealed by Registration of Multiple Modalities"](https://www.biorxiv.org/content/10.1101/2023.07.27.550885v1)

In this paper we show how the systematic importance of registration for finding genetic signals directly from medical imaging modalities.
This is demonstrated across a wide array of registration techniques.  
Our multimodal autoencoder comparison framework allows us to learn representations of medical images before and after registration.
The learned registration methods considered are graphically summarized here:
![Learned Registration Methods](./registration.png)

For example, to train a uni-modal autoencoder for DXA 2 scans:
```bash
  python /path/to/ml4h/ml4h/recipes.py \
    --mode train \
    --tensors /path/to/hd5_tensors/ \
    --output_folder /path/to/output/ \
    --tensormap_prefix ml4h.tensormap.ukb \
    --input_tensors dxa.dxa_2 --output_tensors dxa.dxa_2 \
    --encoder_blocks conv_encode --merge_blocks --decoder_blocks conv_decode \
    --activation swish --conv_layers 32 --conv_width 31 --dense_blocks 32 32 32 32 32 --dense_layers 256 --block_size 3 \
    --inspect_model --learning_rate 0.0001 \
    --batch_size 4 --epochs 216 --training_steps 128 --validation_steps 36 --test_steps 4 --patience 36 \
    --id dxa_2_autoencoder_256d
```

To train the cross-modal (DXA 2 <-> DXA5) registration with the DropFuse model the command line is:
```bash
  python /path/to/ml4h/ml4h/recipes.py \
    --mode train \
    --tensors /path/to/hd5_tensors/ \
    --output_folder /path/to/output/ \
    --tensormap_prefix ml4h.tensormap.ukb \
    --input_tensors dxa.dxa_2 dxa.dxa_5 --output_tensors dxa.dxa_2 dxa.dxa_5 \
    --encoder_blocks conv_encode --merge_blocks pair --decoder_blocks conv_decode \
    --pairs dxa.dxa_2 dxa.dxa_5 --pair_loss contrastive --pair_loss_weight 0.1 --pair_merge dropout \
    --activation swish --conv_layers 32 --conv_width 31 --dense_blocks 32 32 32 32 32 --dense_layers 256 --block_size 3 \
    --inspect_model --learning_rate 0.0001 \
    --batch_size 4 --epochs 216 --training_steps 128 --validation_steps 36 --test_steps 4 --patience 36 \
    --id dxa_2_5_dropfuse_256d
```
Similiarly, autoencoders and cross modal fusion for all the modalities considered in the paper can be trained by changing the `--input_tensors` and `--output_tensors` arguments to point at the appropriate `TensorMap`, and if necessary updating the model architecture hyperparameters.
Table 1 lists all the modalities included in the paper.
![Table of modalities](./table1.png)

Then with latent space inference with models before and after registration we can evaluate their learned representations.
```bash
    python /home/sam/ml4h/ml4h/recipes.py \
    --mode infer_encoders \
    --tensors /path/to/hd5_tensors/ \
    --output_folder /path/to/output/ \
    --tensormap_prefix ml4h.tensormap.ukb \
    --input_tensors dxa.dxa_2 --output_tensors dxa.dxa_2 \ 
    --id dxa_2_autoencoder_256d \
    --model_file /path/to/output/dxa_2_autoencoder_256d/dxa_2_autoencoder_256d.h5
```

We compare the strength and number of biological signals found with the [Latent Space Comparisons notebook](./latent_space_comparisons.ipynb).
This notebook is used to populate the data summarized in Table 2 of the paper.
![Table of results](./table2.png)