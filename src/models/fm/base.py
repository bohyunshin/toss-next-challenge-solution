from abc import abstractmethod
from typing import List

import torch
from torch import nn


class BaseFM(nn.Module):
    def __init__(
        self,
        categorical_field_dims: List[int] = None,
        numerical_field_count: int = 0,
        **kwargs,
    ):
        super(BaseFM, self).__init__()

        self.categorical_field_dims = categorical_field_dims or []
        self.numerical_field_count = numerical_field_count
        self.num_categorical = len(self.categorical_field_dims)
    
    @abstractmethod
    def forward(**kwargs):
        raise NotImplementedError
    
    def _setup_categorical_embeddings(self):
        """Setup categorical embeddings for second-order interactions"""
        # Reuse the same total_vocab_size from parent class
        total_vocab_size = self.field_offsets[-1]
        self.categorical_embeddings = nn.Embedding(total_vocab_size, self.embed_dim)

    def _setup_numerical_embeddings(self):
        """Setup numerical embeddings for second-order interactions"""
        # Each numerical feature gets its own embed_dim-dimensional latent vector
        self.numerical_embeddings = nn.Parameter(
            torch.randn(self.numerical_field_count, self.embed_dim)
        )
    
    def _init_embedding_weights(self):
        """Initialize embedding weights for second-order interactions"""
        if hasattr(self, "categorical_embeddings"):
            nn.init.xavier_normal_(self.categorical_embeddings.weight, gain=1.0)

        if hasattr(self, "numerical_embeddings"):
            nn.init.xavier_normal_(self.numerical_embeddings, gain=1.0)
    
    def _get_all_embeddings(self, numerical_x, categorical_x, seq_emb = None, is_num_weighted = False):
        """Get all embeddings for CIN (same as DeepFM)"""
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        all_embeddings = []

        # Categorical embeddings
        if categorical_x is not None and self.num_categorical > 0:
            global_indices = categorical_x + self.field_offsets_tensor.unsqueeze(0)
            cat_embeddings = self.categorical_embeddings(global_indices)
            all_embeddings.append(cat_embeddings)

        # Numerical embeddings
        if numerical_x is not None and self.numerical_field_count > 0:
            num_embeddings = self.numerical_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            if is_num_weighted:
                numerical_x_expanded = numerical_x.unsqueeze(-1)
                num_embeddings = num_embeddings * numerical_x_expanded
            all_embeddings.append(num_embeddings)
        
        # concat sequence embeddings if specified
        if seq_emb:
            all_embeddings.append(seq_emb.unsqueeze(1))

        if not all_embeddings:
            return None

        # Concatenate all embeddings
        embeddings = torch.cat(
            all_embeddings, dim=1
        )  # (batch_size, total_fields, embed_dim)

        return embeddings
    
    def _get_all_x_values(self, numerical_x, categorical_x, use_seq_emb = False):
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        device = next(self.parameters()).device

        # Collect all embeddings and values in tensors (fully vectorized)
        all_x_values = []

        # Categorical embeddings and values
        if categorical_x is not None and self.num_categorical > 0:
            # For categorical features, x_i = 1
            cat_x_values = torch.ones(
                batch_size, self.num_categorical, 1, device=device, dtype=torch.float32
            )
            all_x_values.append(cat_x_values)

        # Numerical embeddings and values
        if numerical_x is not None and self.numerical_field_count > 0:
            num_x_values = numerical_x.unsqueeze(-1)
            all_x_values.append(num_x_values)
        
        # if using seq embedding, concat one vector
        if use_seq_emb:
            all_x_values.append(
                torch.ones(batch_size, 1, 1, device=device, dtype=torch.float32)
            )
        
        return torch.cat(all_x_values, dim=1)  # (batch_size, total_features, 1)