import torch
import torch.nn as nn
import numpy as np
from tensorflow import keras
from performer_pytorch import Performer


class ECGKerasEncoderWrapper(nn.Module):
    def __init__(self, transformer_dim=64, num_layers=4, num_heads=4, dropout=0.1,
                 keras_model_path="/home/rrathod3/runs/ecg2age_v2025_09_11/ecg2age_v2025_09_11.keras"):
        super().__init__()

        # Load keras encoder
        keras_model = keras.models.load_model(keras_model_path, compile=False)
        encoder_output = keras_model.get_layer("activation_16").output
        self.keras_encoder = keras.Model(inputs=keras_model.input, outputs=encoder_output)

        '''
        # Transformer (longitudinal)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        '''
        self.transformer = Performer(
            dim=transformer_dim,               # hidden size
            depth=num_layers,                  # number of layers
            heads=num_heads,                   # number of attention heads
            dim_head=transformer_dim // num_heads,
            ff_mult=4,                         # feedforward expansion factor
            causal=False                       # not autoregressive
        )

        self.reg_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, 1)
        )

    def forward(self, ecg_seq, mask=None):
        """
        ecg_seq: [B, SeqLen, 5000, 12]
        mask:    [B, SeqLen] (bool, 1=real, 0=padded)
        """
        B, L, T, C = ecg_seq.shape
        device = ecg_seq.device

        # Flatten visits
        flat = ecg_seq.reshape(B * L, T, C).cpu().numpy()

        # Encode each ECG separately
        features_np = self.keras_encoder(flat, training=False).numpy()  # [B*L, PatchLen, D]
        z = torch.from_numpy(features_np).float().to(device)

        # Pool across PatchLen → per-visit embedding
        z = z.mean(dim=1)  # [B*L, D]

        # Reshape back to [B, SeqLen, D]
        z = z.view(B, L, -1)
        
        '''
        # Pass through Transformer (with mask)
        if mask is not None:
            # Transformer expects mask in shape [B, L], where False=keep, True=ignore
            # PyTorch uses src_key_padding_mask with opposite convention
            z = self.transformer(z, src_key_padding_mask=~mask)
        else:
            z = self.transformer(z)
        '''
        z = self.transformer(z)
        # Mean pool over valid visits
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, L, 1]
            z = (z * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            z = z.mean(dim=1)

        return self.reg_head(z).squeeze(-1)  # [B]
