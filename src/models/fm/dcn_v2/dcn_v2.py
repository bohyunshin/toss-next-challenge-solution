import torch
from models.fm.dcn_v2.dcn_v2_base import DCNv2Base


class Model(DCNv2Base):
    """
    Deep Cross Network V2 (DCN-v2)

    Improvements over DCN-v1:
    - Multiple architecture options (parallel, stacked, stacked_parallel)
    - Enhanced cross network with low-rank matrix decomposition
    - Better model capacity and gradient flow
    - Optional mixture of experts in cross layers
    """

    def __init__(
        self,
        categorical_field_dims=None,
        numerical_field_count=0,
        embed_dim=10,
        cross_layers=2,
        deep_layers=[256, 128, 64],
        dropout_rate=0.2,
        structure="stacked",  # 'parallel', 'stacked', 'stacked_parallel'
        use_low_rank=True,
        low_rank=32,
        num_experts=1,
        **kwargs,
    ):
        # Initialize parent class
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
            cross_layers=cross_layers,
            deep_layers=deep_layers,
            dropout_rate=dropout_rate,
            structure=structure,
            use_low_rank=use_low_rank,
            low_rank=low_rank,
            num_experts=num_experts,
            use_seq_features=False,
            **kwargs,
        )

    def forward(self, numerical_x=None, categorical_x=None, **kwargs):
        """
        Forward pass of DCN-v2 model

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            DCN-v2 predictions (batch_size, 1)
        """
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )
        
        numerical_x = self.bn_num(numerical_x)

        # Create dense input vector
        dense_input = self._get_all_embeddings(numerical_x, categorical_x, is_num_weighted = True).view(batch_size, -1)

        if self.structure == "parallel":
            # Original DCN-v1 style (parallel cross and deep)
            cross_output = self.cross_network(dense_input)
            deep_output = self.deep_network(dense_input)
            combined_output = torch.cat([cross_output, deep_output], dim=1)

        elif self.structure == "stacked":
            # Cross network first, then deep network
            cross_output = self.cross_network(dense_input)
            combined_output = self.deep_network(cross_output)

        elif self.structure == "stacked_parallel":
            # Both stacked and parallel paths
            cross_output = self.cross_network(dense_input)
            deep_stacked_output = self.deep_network(cross_output)
            deep_parallel_output = self.deep_network_parallel(dense_input)

            combined_output = torch.cat(
                [cross_output, deep_stacked_output, deep_parallel_output], dim=1
            )

        else:
            raise ValueError(f"Unknown structure: {self.structure}")

        # Final prediction
        output = self.output_layer(combined_output)
        return output
