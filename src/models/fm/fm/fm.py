import torch
from models.fm.fm.fm_base import FMBase


class Model(FMBase):
    """
    Factorization Machine inheriting from LogisticRegression

    FM formula: ŷ = w₀ + Σwᵢxᵢ + Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ

    Inherits first-order interactions from LogisticRegression,
    adds second-order interactions via embeddings.
    """

    def __init__(
        self,
        categorical_field_dims=None,
        numerical_field_count=0,
        embed_dim=10,
        **kwargs,
    ):
        # Initialize parent class (gets bias + first-order interactions)
        super().__init__(
            categorical_field_dims=categorical_field_dims,
            numerical_field_count=numerical_field_count,
            embed_dim=embed_dim,
        )

    def forward(self, numerical_x=None, categorical_x=None, **kwargs):
        """
        Forward pass of FM: LR + second-order interactions

        Args:
            categorical_x: Categorical features (batch_size, num_categorical)
            numerical_x: Numerical features (batch_size, num_numerical)

        Returns:
            FM predictions (batch_size, 1)
        """
        numerical_x = self.bn_num(numerical_x)

        # Get first-order interactions from parent class
        batch_size = (
            categorical_x.size(0) if categorical_x is not None else numerical_x.size(0)
        )

        # Start with bias term (w₀)
        output = self.bias.expand(batch_size).clone()

        # Add first-order interactions (Σwᵢxᵢ)
        output += self._first_order_interactions(numerical_x, categorical_x)

        # Add second-order interactions
        output += self._second_order_interactions(numerical_x, categorical_x)

        return output.unsqueeze(-1)  # (batch_size, 1)

