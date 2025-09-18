from typing import List, Optional

import torch
import torch.nn as nn

from utils import init_weights


class SingleInputEmbedding(nn.Module):
    """
    Embeds a single continuous input into a higher dimensional feature space.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 dropout: float = 0.1) -> None:
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel)
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):
    """
    Embeds multiple continuous (and optionally categorical) inputs
    into a shared feature space and aggregates them.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 dropout: float = 0.1) -> None:
        super(MultipleInputEmbedding, self).__init__()
        
        if not in_channels:
            raise ValueError("in_channels cannot be empty")
        
        self.num_inputs = len(in_channels)
        self.out_channel = out_channel
        
        self.module_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(out_channel, out_channel)
            )
            for in_channel in in_channels
        ])
        
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel)
        )
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        
        # Input validation
        if len(continuous_inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} continuous inputs, got {len(continuous_inputs)}")
        
        # Process continuous inputs
        embedded_outputs = []
        for i, input_tensor in enumerate(continuous_inputs):
            if input_tensor.size(-1) == 0:  # Skip empty tensors
                continue
            embedded_outputs.append(self.module_list[i](input_tensor))
        
        if not embedded_outputs:
            raise ValueError("No valid continuous inputs to process")
            
        # Aggregate continuous embeddings
        output = torch.stack(embedded_outputs, dim=0).sum(dim=0)

        # Add categorical inputs if provided
        if categorical_inputs is not None:
            categorical_outputs = [cat_input for cat_input in categorical_inputs if cat_input.size(-1) > 0]
            if categorical_outputs:
                categorical_sum = torch.stack(categorical_outputs, dim=0).sum(dim=0)
                # Ensure same shape for addition
                if categorical_sum.shape != output.shape:
                    raise ValueError(f"Categorical input shape {categorical_sum.shape} doesn't match continuous output shape {output.shape}")
                output = output + categorical_sum

        return self.aggr_embed(output)