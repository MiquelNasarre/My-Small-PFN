
## --- LICENSE AND COPYRIGHT ---
## =======================================================================================
##  * My Small PFN — a competitive proof-of concept for Prior-Data Fitted Networks
##  * Copyright (c) 2026 Miguel Nasarre Budiño
##  * Licensed under the MIT License. See LICENSE file.
## =======================================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from dataclasses import dataclass, asdict
from contextlib import nullcontext
from functools import lru_cache
import json


@dataclass
class ModelConfig:
    '''
    Class that stores the hyper-parameters for the creation of the PFN.
    Includes general dimensions and metadata like the embedded dimension,
    number of heads, hidden dimensions, etc.

    It also includes two user-facing knobs: temperature and device.
    '''

    # Whether the model will require gradients or not, can be changed post-init
    requires_grad: bool = False

    # Embedded dimension for the tokens (table cells)
    embedded_dimension: int = 64

    # Number of layers in the transformer
    n_layers: int = 12

    # Number of heads used for attention
    n_heads: int = 2

    # Hidden dimension in the feed-forward MLP
    hidden_dimension_ff: int = 256

    # Hidden dimension used by the encoder
    hidden_dimension_enc: int = 256

    # Hidden dimension used by the decoder
    hidden_dimension_dec: int = 256

    # Number of features to be grouped for tokenization
    feature_group_size: int = 3

    # Thinking rows added to the transformer input
    n_thinking_rows: int = 16

    # Number of buckets to discretize the real number line
    n_buckets: int = 64

    # Temperature adjustment knob
    temperature: float = 1.0

    # Device to be used for inference
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class BucketOps:
    '''
    Since the output of the PFN is not a number but a probability distribution
    across the real line represented by a real line discretization, we need 
    some operations to work with this bucket representations.

    This class is a holder for all the functions related to this representation
    that will help you operate with the model output with ease.
    '''

    @staticmethod
    @lru_cache(maxsize=128)
    def get_bucket_boundaries(n_buckets: int) -> torch.Tensor:
        '''
        Function that returns the bucket boundaries for a subdivision by a given 
        amount of buckets. Defines the discretization.

        This discretization is based on an equal probability for all buckets 
        when the target follows a standard normal distribution.

        Therefore each bucket will have equal probability given a normal distribution
        with mean zero and deviation one.
        
        Args:
            n_buckets: Parameter that determines the number of buckets to be used, 
                normally config.buckets
        '''
        
        # Define the distribution we are working with
        normal = torch.distributions.Normal(loc=0.0, scale=1.0)

        # Create the vector of quantiles
        p = (torch.arange(1, n_buckets) / n_buckets)  # 1/n_buckets ... (n_buckets-1)/n_buckets

        # Get boundaries on the real number line
        edges = normal.icdf(p) # (n_buckets - 1,)

        # Return boundaries
        return edges # (n_buckets - 1,)

    @staticmethod
    def real_to_bucket(n_buckets: int, real_numbers: torch.Tensor | float) -> torch.Tensor | int:
        '''
        Given a tensor of real number this function returns the tensor of same 
        shape with the bucket label that each number corresponds to.

        To be used when transforming the test outputs into their corresponding 
        labels.
        
        Args:
            n_buckets: Parameter that determines the number of buckets to be used, 
                normally config.buckets

            real_numbers: Tensor or float from which the buckets will be inferred.
        '''

        # Get the boundaries
        edges = BucketOps.get_bucket_boundaries(n_buckets)

        # Convert to tensor if needed
        x = torch.as_tensor(real_numbers)

        # Match device
        edges = edges.to(x.device)

        # Bucketize
        buckets = torch.bucketize(x, edges, right=True).to(dtype=torch.long)

        # If single item return integer
        if buckets.ndim == 0:
            buckets = int(buckets.item())

        # Return buckets
        return buckets

    @staticmethod
    def bucket_medians(n_buckets: int, buckets: torch.Tensor | int) -> torch.Tensor | float:
        '''
        Given a list of buckets it tells you the median position of each bucket.
        The median being the position that sits at equal probability in either 
        side of the bucket.

        Does not check for out of bound buckets, always provide correct indices.
        
        Args:
            n_buckets: Parameter that determines the number of buckets to be used, 
                normally config.buckets

            buckets: Integer tensor or integer labeling the buckets 
                to compute the median for.
        '''

        # Convert to tensor if needed
        buckets = torch.as_tensor(buckets)

        # Define the distribution we are working with
        normal = torch.distributions.Normal(loc=0.0, scale=1.0)

        # Obtain mid-probabilities of buckets
        probs = (buckets + 0.5) / n_buckets

        # Get mid positions for those buckets
        mids = normal.icdf(probs) # (n_buckets - 1,)

        # If single item return floating point
        if mids.ndim == 0:
            mids = mids.item()

        # Return middle points
        return mids
    
    @staticmethod
    def bucket_means(n_buckets: int, buckets: torch.Tensor | int) -> torch.Tensor | float:
        '''
        Given a list of buckets it tells you the mean position of each bucket.
        The mean of a bucket being the expected value of Z given Z is inside it.

        Does not check for out of bound buckets, always provide correct indices.
        
        Args:
            n_buckets: Parameter that determines the number of buckets to be used, 
                normally config.buckets

            buckets: Integer tensor or integer labeling the buckets 
                to compute the mean for.
        '''

        # Convert to tensor if needed
        buckets = torch.as_tensor(buckets)

        # Bucket boundaries
        edges = BucketOps.get_bucket_boundaries(n_buckets).to(buckets.device)

        # Build a and b for each bucket index
        a = torch.empty_like(buckets, dtype=torch.float32)
        b = torch.empty_like(buckets, dtype=torch.float32)

        # Left boundary for wach bucket index
        a[buckets == 0] = -torch.inf
        a[buckets > 0] = edges[buckets[buckets > 0] - 1]

        # Right boundary for each bucket index
        b[buckets < n_buckets - 1] = edges[buckets[buckets < n_buckets - 1]]
        b[buckets == n_buckets - 1] = torch.inf

        # PDF and CDF
        phi = lambda x: torch.exp(-0.5 * x * x) / (2 * torch.pi)**0.5
        Phi = lambda x: 0.5 * (1.0 + torch.erf(x / 2**0.5))

        # Calculate bucket means
        means = (phi(a) - phi(b)) / (Phi(b) - Phi(a))

        # If single item return floating point
        if means.ndim == 0:
            means = means.item()

        # Return bucket means
        return means

    @staticmethod
    def probs_to_mean_var_std(bucket_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[float, float, float]:
        '''
        Given an ouput bucket probabilities tensor it outputs the mean, variance 
        and standard deviation of the variable it is representing.

        This is done by computing the expectation of Z and the expectation of Z^2.
        Then scaling those probabilities by the bucket probabilities to obtain the 
        different variables. Details can be seen on implementation.
        
        Args:
            bucket_probs: Tensor which its leading dimension contains probability 
                distributions over buckets, usually outputs of the PFN regressor.
        '''

        # Assumes shape bucket_probs: (..., n_buckets)
        n_buckets = bucket_probs.shape[-1]
        # Store device for intermediate tensor operations
        device = bucket_probs.device

        # Boundaries
        edges = BucketOps.get_bucket_boundaries(n_buckets).to(device)  # (n_buckets-1,)

        # Build left and right boundaries for all buckets
        a = torch.empty(n_buckets, device=device, dtype=torch.float32)
        b = torch.empty(n_buckets, device=device, dtype=torch.float32)

        a[0] = -torch.inf
        a[1:] = edges
        b[:-1] = edges
        b[-1] = torch.inf

        # PDF and CDF
        phi = lambda x: torch.exp(-0.5 * x * x) / (2 * torch.pi) ** 0.5
        Phi = lambda x: 0.5 * (1.0 + torch.erf(x / 2 ** 0.5))

        # Z value helper
        Z = Phi(b) - Phi(a) # (n_buckets,)

        # Per bucket mean calculation
        mu_i = (phi(a) - phi(b)) / Z
        
        # Per bucket second moment (handle edges explicitly)
        ez2_i = torch.empty_like(mu_i)
        # Interior buckets
        ez2_i[1:-1] = 1.0 + (a[1:-1] * phi(a[1:-1]) - b[1:-1] * phi(b[1:-1])) / Z[1:-1]
        # Left edge
        ez2_i[0] = 1.0 - b[0] * phi(b[0]) / Phi(b[0])
        # Right edge
        ez2_i[-1] = 1.0 + a[-1] * phi(a[-1]) / (1.0 - Phi(a[-1]))
        
        # Mix moments using bucket_probs (broadcasts over leading dims)
        mean = (bucket_probs * mu_i).sum(dim=-1)
        ez2 = (bucket_probs * ez2_i).sum(dim=-1)

        # Compute variance and standard deviation
        var = ez2 - mean * mean
        std = torch.sqrt(var)

        # If only one dimension provided return single floats
        if bucket_probs.dim() == 1:
            mean = mean.item()
            var = var.item()
            std = std.item()

        # Return mean, var and std
        return mean, var, std
        
    @staticmethod
    def probs_to_distribution_plot(bucket_probs: torch.Tensor, values: torch.Tensor | float) -> torch.Tensor | float:
        '''
        Given an ouput bucket probabilities tensor you might want to recreate the 
        distribution function for different values or maybe plot it. 

        This function takes as input the bucket prbabilities and some values and 
        gives you the distribution output for those values. Since the distribution 
        is discrete, the output is not continuous.
        
        Args:
            bucket_probs: Tensor which its leading dimension contains probability 
                distributions over buckets, usually outputs of the PFN regressor.

            values: Tensor or single value that will be used to output distribution.
        '''

        # Convert to tensor if needed
        bucket_probs = torch.as_tensor(bucket_probs)

        # Assumes shape bucket_probs: (..., n_buckets)
        n_buckets = bucket_probs.shape[-1]
        # Store device for intermediate tensor operations
        device = bucket_probs.device
        values = torch.as_tensor(values, device=device)

        # Obtain the default heights of the values
        values_default_heights = torch.exp(-0.5 * values * values) / (2.0*torch.pi)**0.5

        # Separate values into buckets
        buckets = torch.as_tensor(BucketOps.real_to_bucket(n_buckets, values), device=device, dtype=torch.long) # (values.shape)

        # Calculate the values relative height depending on their bucket probabilities
        lead_shape = bucket_probs.shape[:-1]          # (...) e.g. (B, T)
        idx = buckets.view((1,) * len(lead_shape) + buckets.shape)  # (1,1,..., *x.shape)
        idx = idx.expand(*lead_shape, *buckets.shape)               # (..., *x.shape)

        # Obtain the bucket probability for each value
        values_bucket_height = torch.gather(bucket_probs, dim=-1, index=idx) * n_buckets  # (..., *x.shape)

        # Multiply to obtain the final height
        values_height = values_default_heights * values_bucket_height

        # Scalar niceness
        if values_height.ndim == 0:
            return float(values_height.item())

        # Return the values heights
        return values_height # (..., values.shape)
        

class MultiHeadSelfAttention(nn.Module):
    '''
    Simple MHSA module used by each Layer for both attention passes.

    Dimensions are specified by Layer given the ModelConfig.
    '''

    def __init__(self, n_heads: int, emb_dim: int):
        super().__init__()

        # Dimensions sanity check
        if emb_dim % n_heads != 0:
            raise RuntimeError("The embedded dimension must be divisible by the number of heads.")
        
        # Store dimensions
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads

        # Create QKV projection
        self.linear_qkv = nn.Linear(emb_dim, 3 * emb_dim)

        # Create Output projection
        self.linear_out = nn.Linear(emb_dim, emb_dim)

    def forward(self, attn_in: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        
        # Store initial shape
        N, L, E = attn_in.shape # (N, L, emb_dim)
        
        # Obtain Q, K, V matrix from same input
        qkv = self.linear_qkv(attn_in) # (N, L, 3 * emb_in)
        Q, K, V = qkv.chunk(3, dim=-1) # (N, L, emb_in) each

        # Reshape and transpose for multiple heads (emb_dim = h * dH)
        Q = Q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2) # (N, h, L, dH)
        K = K.view(N, L, self.n_heads, self.head_dim).transpose(1, 2) # (N, h, L, dH)
        V = V.view(N, L, self.n_heads, self.head_dim).transpose(1, 2) # (N, h, L, dH)

        # Use flash attention from torch
        O = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, dropout_p=0.0)

        # Reshape it back to normal 
        O = O.transpose(1, 2).contiguous().view(N, L, E)

        # Apply output linear
        attn_out = self.linear_out(O) # (N, L, emb_dim)

        # Return output
        return attn_out # (N, L, emb_dim)


class FeedForward(nn.Module):
    '''
    Simple MLP used by each Layer of the Transformer after the double 
    attention. 
    
    Dimensions are specified by Layer given the ModelConfig.
    '''

    def __init__(self, embedded_dim: int, hidden_dim: int):
        super().__init__()

        # MLP that will perform the feed-forward step
        self.mlp = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedded_dim)
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # Do feed-forward step
        return self.mlp(emb) # (B, S+T, Fg+1, emb_dim)
    

class Layer(nn.Module):
    '''
    Single layer of the PFN transformer, consists of two attention blocks,
    and a feed-forward layer with some normalization steps.

    The forward pass first does feature attention, then transposes and does
    row attention with a mask to prevent test attention, and then transposes 
    back and does the feed forward step. 

    This closely mimics the architecture used in TabPFN-2.5.

    Dimensions are specified by the Transformer given the ModelConfig.
    '''
    
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Define attention on each row between features
        self.feature_attn = MultiHeadSelfAttention(config.n_heads, config.embedded_dimension)
        self.layer_norm_fa = nn.LayerNorm(config.embedded_dimension)

        # Define attention on each feature between rows
        self.row_attn = MultiHeadSelfAttention(config.n_heads, config.embedded_dimension)
        self.layer_norm_ra = nn.LayerNorm(config.embedded_dimension)

        # Define feed forward step
        self.feed_forward = FeedForward(config.embedded_dimension, config.hidden_dimension_ff)
        self.layer_norm_ff = nn.LayerNorm(config.embedded_dimension)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, ST, Fg1, E = emb.shape # (B, S+T, Fg+1, emb_dim)

        # Reshape to create feature attention input
        f_attn_in = emb.reshape([B*ST, Fg1, E]) # (B*(S+T), Fg+1, emb_dim)
        # Run feature attention
        f_attn_out = self.feature_attn(f_attn_in) # (B*(S+T), Fg+1, emb_dim)
        # Add residual and layer normalize
        res_f_attn_out = self.layer_norm_fa(f_attn_out + f_attn_in) # (B*(S+T), Fg+1, emb_dim)

        # Reshape and transpose to create row attention input
        r_attn_in = res_f_attn_out.reshape([B, ST, Fg1, E]).transpose(1,2).contiguous().reshape([B*Fg1, ST, E]) # (B*(Fg+1), S+T, emb_dim)
        # Run row attention
        r_attn_out = self.row_attn(r_attn_in, mask) # (B*(Fg+1), S+T, emb_dim)
        # Add residual and layer normalize
        res_r_attn_out = self.layer_norm_ra(r_attn_out + r_attn_in) # (B*(Fg+1), S+T, emb_dim)

        # Reshape and transpose back to original shape for feed-forward input
        ff_in = res_r_attn_out.reshape([B, Fg1, ST, E]).transpose(1,2).contiguous() # (B, S+T, Fg+1, emb_dim)
        # Run feed-forward
        ff_out = self.feed_forward(ff_in) # (B, S+T, Fg+1, emb_dim)
        # Add residual and layer normalize
        layer_out = self.layer_norm_ff(ff_out + ff_in) # (B, S+T, Fg+1, emb_dim)

        # Return layer output
        return layer_out # (B, S+T, Fg+1, emb_dim)


class Transformer(nn.Module):
    '''
    PFN transformer architecture. Simple stack of transformer layers executed 
    sequentially. 
    
    Dimensions are specified by the PFN given the ModelConfig.
    '''
    
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Create all the layers for the transformer
        self.layers = nn.ModuleList([
            Layer(config)
            for _ in range(config.n_layers)
        ])

    def forward(self, trans_in: torch.Tensor, test_size: int) -> torch.Tensor:

        # Create attention mask for row attention (test attends to train, train attends to train)
        mask = torch.zeros([trans_in.shape[1], trans_in.shape[1]], dtype=trans_in.dtype, device=trans_in.device)
        mask[:,-test_size:] = float('-inf')

        # Iterate through the layers and return
        for layer in self.layers:
            trans_in = layer(trans_in, mask) 
        return trans_in # (B, S+T, Fg+1, emb_dim)


class EncoderX(nn.Module):
    '''
    Encoder for the feature tensor of the PFN. Contains a basic MLP 
    for the embedding.

    The forward pass splits into feature groups, padding with empty 
    features and scaling if necessary, then applies the encoding to 
    output the embedded tokens.

    Dimensions are specified by the PFN given the ModelConfig.
    '''

    def __init__(self, group_size: int, hidden_dim: int, embedded_dim: int):
        super().__init__()

        # Store the group size for later group rearrangement
        self.group_size = group_size

        # MLP to be applied to the feature groups to encode them
        self.mlp = nn.Sequential(
            nn.Linear(group_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedded_dim)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        # Store initial shape
        B, S, F = X.shape # (B, S, F)

        # Append missing columns for feature group creation
        residue = F % self.group_size
        if residue != 0:
            X = torch.cat([X, torch.zeros([B, S, self.group_size - residue], dtype=torch.float32, device=X.device)], dim=2) # (B, S, F_pad)

            # Scale to keep overall group feature density given empty rows
            X[:, :,-self.group_size:] *= (self.group_size / residue) ** 0.5

        # Rearrange into feature groups
        Fg = X.shape[2] // self.group_size
        X = X.view(B, S, Fg, self.group_size) # (B, S, Fg, group_size)

        # Apply the MLP to bring tokens to embedded dimensions
        return self.mlp(X) # (B, S, Fg, emb_dim)
    

class EncoderY(nn.Module):
    '''
    Encoder for the target tensor of the PFN. Contains a basic MLP 
    for the embedding.

    The forward pass appends the test rows, creates the mask to encode
    them and concatenates it to the input, then applies the encoding 
    to output the embedded target tokens.

    Dimensions are specified by the PFN given the ModelConfig.
    '''

    def __init__(self, hidden_dim: int, embedded_dim: int):
        super().__init__()

        # MLP to be applied to the target plus mask to encode them
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedded_dim)
        )

    def forward(self, y: torch.Tensor, test_size: int) -> torch.Tensor:

        # Store initial shape
        B, train_size, _ = y.shape # (B, train_size, 1)

        # Append missing rows to y
        y = torch.cat([y, torch.zeros([B, test_size, 1], dtype=torch.float32, device=y.device)], dim=1) # (B, S, 1)

        # Create missing targets tensor
        mask = torch.zeros_like(y) # (B, S, 1)
        # Missing target is encoded as 1s
        mask[:,-test_size:,0] = 1

        # concatenate tensors
        y_plus_mask = torch.cat([y,mask], dim=2) # (B, S, 2)

        # Apply the MLP to bring target tokens to embedded dimensions and reshape to match features
        return self.mlp(y_plus_mask).view([B, train_size + test_size, 1, -1]) # (B, S, 1, emb_dim)


class Decoder(nn.Module):
    '''
    Decoder to create the output logits of the PFN. Contains a basic MLP 
    for the decoding.

    The forward pass extracts the test target tokens that will be used, 
    then applies the MLP to obtain the output logits for the buckets.

    Dimensions are specified by the PFN given the ModelConfig.
    '''

    def __init__(self, embedded_dim: int, hidden_dim: int, n_buckets: int):
        super().__init__()

        # MLP to be applied to the embedded targets to turn them to logits
        self.mlp = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_buckets)
        )

    def forward(self, trans_out: torch.Tensor, test_size: int) -> torch.Tensor:
        
        # Take the last tokens of the test rows (originally the unknown targets)
        target_tokens = trans_out[:,-test_size:,-1,:] # (B, test_size, emb_dim)

        # Apply the MLP to the tokens to obtain the raw logits
        return self.mlp(target_tokens) # (B, test_size, n_buckets)


class AddThinkingTokens(nn.Module):
    '''
    Simple module mimicking the module used to add the thinking 
    rows in the TabPFN-2.5 architecture.

    Stores a learnable thinking token for each row. To be 
    broadcasted to its entire thinking row and then concatenated 
    to the input tensor during the forward pass.

    Dimensions are specified by the PFN given the ModelConfig.
    '''

    def __init__(self, n_thinking_rows: int, embedded_dim: int):
        super().__init__()

        # store number of rows
        self.n_thinking_rows = n_thinking_rows

        # If no rows do nothing
        if not n_thinking_rows:
            return

        # We have to work with variable numbers of features, 
        # so we use the same token for each feature.
        self.row_token_values = nn.Parameter(torch.empty(n_thinking_rows, embedded_dim)) # (T, emb_dim)

        # This is the initialisation used in torch.nn.Embedding, 
        # so hopefully a reasonable choice for our application.
        torch.nn.init.normal_(self.row_token_values)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:

        # If no rows do nothing
        if not self.n_thinking_rows:
            return emb

        # Get shape information
        B, _, Fg1, _ = emb.shape # (B, S, Fg1, emb_dim)

        # Expand thinking tokens to match shape
        thinking_tokens_base = self.row_token_values.unsqueeze(0).unsqueeze(2) # (1, T, 1, emb_dim)
        thinking_tokens = thinking_tokens_base.expand(B, -1, Fg1, -1)          # (B, T, Fg1, emb_dim)

        # Concatenate
        emb = torch.cat([thinking_tokens, emb], dim=1) # (B, S+T, Fg1, emb_dim)

        # Return
        return emb # (B, S+T, Fg1, emb_dim)
    

class MyRegressorPFN(nn.Module):
    '''
    This is my take on creating a PFN for regression, heavily inspired by 
    the TabPFN approach but with a much smaller scale and without a lot 
    of preprocessing or post-processing steps.

    The forward pass is eliminated, instead fit() and predict() are to be 
    used for inference. fit() is just a holder for the tensors, then predict()
    triggers the entire forward pass, including: preprocessing -> encoders -> 
    transformer -> decoder -> output.

    For creation it expects a model config, if not provided it uses the 
    default config settings.

    It can output either 'logits', 'probs'/'proba'/'probabilities' (over buckets), 
    or 'values' (mean/var/std) or just 'mean'.
    '''

    def __init__(self, model: str | None = None, model_config: ModelConfig | None = None):
        '''
        Constructor for `MyRegressorPFN`. It generates all the necessary `nn.Module`'s 
        to run the forward pass as set by `ModelConfig` and sends them to the 
        specified device.

        Expects a model string or a model configuration. If none is provided it will 
        assume default configuration and generate the model.

        If a model string is provided it will locate the configuration and saved weights 
        inside the `weights` folder and will load the model as requested and set it on 
        inference mode.

        If a `model_config` is provided it will create the model as specified by it, no 
        weights will be loaded. Check `ModelConfig` class for more details.
        '''        
        super().__init__()

        if model is not None and model_config is not None:
            print("WARNING: MyRegressorPFN initializer has received both model and model_config. model_config will be ignored.")

        if model is not None:
            # Get full path
            from pathlib import Path
            # Directory where weights live (for proper model loading)
            WEIGHTS_ = Path(__file__).resolve().parent.parent / "weights"

            # Load the config
            with open(WEIGHTS_ / model / "my_PFN_config.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.config = ModelConfig(**data)
                self.config.requires_grad = False

            # Print
            print("Model config' correctly loaded from", WEIGHTS_ / model / "my_PFN_config.json")

        else:
            # Store the model config, if doesn't exits make default
            self.config = model_config if model_config is not None else ModelConfig()

        # Create the x-encoder
        self.encoder_x = EncoderX(
            self.config.feature_group_size, 
            self.config.hidden_dimension_enc, 
            self.config.embedded_dimension
        )

        # Create the y-encoder
        self.encoder_y = EncoderY(
            self.config.hidden_dimension_enc,
            self.config.embedded_dimension
        )

        # Create the thinking rows module
        self.add_thinking_tokens = AddThinkingTokens(
            self.config.n_thinking_rows, 
            self.config.embedded_dimension
        )

        # Create the transformer stack
        self.transformer = Transformer(
            self.config
        )

        # Create the decoder
        self.decoder = Decoder(
            self.config.embedded_dimension,
            self.config.hidden_dimension_dec,
            self.config.n_buckets
        )

        # Load weights if specific model
        if model is not None:
            self.load_weights(WEIGHTS_ / model / "my_PFN_weights.pth")

        # Send to chosen device
        self.to(self.config.device)

    def save_config_json(self, save_path: str) -> MyRegressorPFN:
        '''
        Saves the `ModelConfig` of the PFN (`self.config`) as a **.json** file.

        `save_path` expects a valid system path to create and save the file.

        Returns itself for convenience.
        '''
        # Create the config file and write your dict
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f)

        # Return self for convenience
        return self

    def load_weights(self, load_path: str) -> MyRegressorPFN:
        '''
        Loads the weights from a previously saved model with the same ModelConfig.

        `load_path` expects a valid path to a file to load the weights from.

        Returns itself for convenience.
        '''
        # Load the state dict
        state = torch.load(load_path, weights_only=True)

        # Remove torch.compile wrapper prefix
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

        # Load weights into the model
        self.load_state_dict(state)

        # Print
        print("Model weights correctly loaded from", load_path)

        # Return itself for simplicity
        return self

    def set_device(self, device: str) -> MyRegressorPFN:
        '''
        Sends the model to the specified torch device given by the input string.

        First sets the device on the `ModelConfig` and then sends the model to it.

        Returns itself for convenience.
        '''
        # Set the device on the model config
        self.config.device = device
        # Send the modules to device
        self.to(device)
        # Return itself for convenience
        return self

    def forward(self):
        raise RuntimeError("This Module does not support forward passes, use fit() and predict() instead.")

    @staticmethod
    def reshape_concatenate_pre_encoder(
            device: str, 
            X_train: torch.Tensor, 
            y_train: torch.Tensor, 
            X_test: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, int]:
        '''
        Internal preprocessing function for the `predict()` forward pass. Reshapes the 
        tensors to the expected shapes and concatenates them to get them ready for the 
        encoders. Not intended to be used by the user directly.

        Most of the model housekeeping is performed here to avoid clogging the forward 
        pass and associated classes with unnecessary checks.

        Returns the `X` and `y` tensors ready for the encoders and the test size.
        '''
        # Make sure your data are torch tensors
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device) # (B?, train_size, F?)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device) # (B?, train_size, 1?)
        X_test  = torch.as_tensor(X_test,  dtype=torch.float32, device=device) # (B?, test_size?, F?)

        # Reshape X_train to match format (B, train_size, F)
        if X_train.dim() == 2:
            # For inference simplicity we will assume batch size to be 1
            # If you want multiple batches you will need to provide the full format
            X_train = X_train.unsqueeze(dim=0)
        elif X_train.dim() == 1:
            # For inference simplicity we will assume batch size is 1 and there is only 1 feature
            # A single training example case is rare so if that is the format it should be explicitly 
            # provided at least as a tensor of shape (1, F)
            X_train = X_train.unsqueeze(dim=0).unsqueeze(dim=-1)
        elif X_train.dim() != 3:
            # Sanity check
            raise RuntimeError(
                "Mismatch in shape of tensors found during preprocessing." \
                "X_train can only have 1, 2 or 3 dimensions."
            )

        # Store shape
        B, train_size, F = X_train.shape

        # Reshape y_train to match format (B, train_size, 1)
        if y_train.dim() == 3:
            # Dimensions must exactly match
            if y_train.shape[0] != B or y_train.shape[1] != train_size or y_train.shape[2] != 1:
                raise RuntimeError(
                    "Mismatch in shape of tensors found during preprocessing." \
                    "If X_train is of shape (B,train_size,F), a 3 dimensional y_train must exactly match the format (B, train_size, 1)."
                )
        elif y_train.dim() == 2:
            if B > 1:
                # Dimensions must be (B, train_size)
                if y_train.shape[0] != B or y_train.shape[1] != train_size:
                    raise RuntimeError(
                        "Mismatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1, a 2 dimensional y_train must exactly match the format (B, train_size/1)."
                    )
                y_train = y_train.unsqueeze(dim=-1) # (B, train_size, 1)
            else: # train_size must match and the other dimension should be 1
                if y_train.shape[0] == train_size and y_train.shape[1] == 1:
                    y_train = y_train.unsqueeze(dim=0) # (B=1, train_size, 1)
                elif y_train.shape[0] == 1 and y_train.shape[1] == train_size:
                    y_train = y_train.unsqueeze(dim=-1) # (B=1, train_size, 1)
                else:
                    raise RuntimeError(
                        "Mismatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size 1, a 2 dimensional y_train must be of shape (train_size, 1) or (1, train_size)."
                    )
        elif y_train.dim() == 1:
            if B > 1:
                if y_train.shape[0] != B:
                    raise RuntimeError(
                        "Mismatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1, y_train must match the batch size."
                    )
                # train_size can not be more than 1
                if train_size > 1:
                    raise RuntimeError(
                        "Mismatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size and train_size bigger than 1, y_train can not be one dimensional."
                    )
                # Assume format (B,)
                y_train = y_train.view([B, 1, 1]) # (B, train_size=1, 1)
            else: # train_size must match
                if y_train.shape[0] != train_size:
                    raise RuntimeError(
                        "Mismatch in shape of tensors found during preprocessing." \
                        "If X_train has train_size bigger than 1, y_train must contain the same train_size."
                    )
                # Assume format (train_size,)
                y_train = y_train.view([1, train_size, 1]) # (B=1, train_size, 1)
        else:
            # Sanity check
            raise RuntimeError(
                "Mismatch in shape of tensors found during preprocessing." \
                "y_train can only have 1, 2 or 3 dimensions."
            )

        # Reshape X_test tp match format (B, test_size, F)
        if X_test.dim() == 3:
            if X_test.shape[0] != B or X_test.shape[2] != F:
                raise RuntimeError(
                    "Mismatch in shape of tensors found during preprocessing." \
                    "If X_train is of shape (B,train_size,F), X_test with 3 dimensions must match B and F."
                )
        elif X_test.dim() == 2:
            if B > 1:
                # Batch size must match on both tensors
                if X_test.shape[0] != B:
                    raise RuntimeError(
                        "Mismatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1, X_test must match the batch size."
                    )
                if F > 1: # Expect format (B, F) with test size 1
                    if X_test.shape[1] == F:
                        X_test = X_test.view([B, 1, F]) # (B, test_size=1, F)
                    else:
                        raise RuntimeError(
                            "Mismatch in shape of tensors found during preprocessing." \
                            "If X_train has a number of features bigger than 1, X_test must match the number of features."
                        )
                else: # Assume format (B, test_size) works if test_size is 1 as well
                    X_test = X_test.view([B, -1, 1]) # (B, test_size, F=1)
            else:
                if X_test.shape[1] == F: # Assume format (test_size, F) works if test_size is 1 and if F is 1 as well
                    X_test = X_test.view([1, -1, F])
                else:
                    if F > 1:
                        raise RuntimeError(
                            "Mismatch in shape of tensors found during preprocessing." \
                            "If X_train has a number of features bigger than 1, X_test must match the number of features."
                        )
                    if X_test.shape[0]>1:
                        raise RuntimeError(
                            "Mismatch in shape of tensors found during preprocessing." \
                            "X_train has 1 feature and batch size 1, if X_test is 2 dimensional at least one dimension must be 1."
                        )
                    # Assume shape (B=1, test_size)
                    X_test = X_test.view([1, -1, 1]) # (B=1, test_size, F=1)
        elif X_test.dim() == 1:
            if B > 1:
                # Batch size must match on both tensors
                if X_test.shape[0] != B:
                    raise RuntimeError(
                        "Mismatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1, X_test must match the batch size."
                    )
                # Feature length can not be more than 1
                if F > 1:
                    raise RuntimeError(
                        "Mismatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1 and more than one feature, X_test can not be one dimensional."
                    )
                # Assume format (B,) and with test_size 1
                X_test = X_test.view([B, 1, 1]) # (B, test_size=1, F=1)
            elif F > 1:
                # Feature length must match
                if X_test.shape[0] != F:
                    raise RuntimeError(
                        "Mismatch in shape of tensors found during preprocessing." \
                        "If X_train has a number of features bigger than 1, X_test must match the number of features."
                    )
                # Assume format (F,) and with test_size 1
                X_test = X_test.view([1, 1, F]) # (B=1, test_size=1, F)
            else: # Assume format (test_size,)
                X_test = X_test.view([1, -1, 1]) # (B=1, test_size, F=1)
        else:
            # Sanity check
            raise RuntimeError(
                "Mismatch in shape of tensors found during preprocessing." \
                "X_test can only have 1, 2 or 3 dimensions."
            )

        # Get test size
        test_size = X_test.shape[1]

        # Concatenate sets
        X = torch.cat([X_train, X_test], dim = -2) # (B, S = train_size + test_size, F)

        # Return preprocessed tensors and test size
        return X, y_train, test_size

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> MyRegressorPFN:
        '''
        Simple data holder function for the training split data to mimic the 
        `fit()`/`predict()` scheme of TabPFN. `fit()`must be called before `predict()`.

        Expects both tensors to have the same amount of training samples, and batch size 
        if this is bigger than one. 

        Other array-like structures are allowed as long as they can be converted to a 
        floating point `torch.Tensor`.

        Returns itself for convenience and to allow `fit()`/`predict()` concatenation.
        '''
        # Store training tensors for inference
        self.X_train = X_train # (B?, train_size, F?)
        self.y_train = y_train # (B?, train_size, 1?)

        # Return itself for simple fit/predict concatenation
        return self

    def predict(self, X_test: torch.Tensor, output: str = 'logits', amp_dtype: torch.dtype | None = torch.bfloat16) -> torch.Tensor:
        '''
        Main method of the class, performs the forward pass on the entire dataset and 
        returns the predictions in the specified format.

        The test data tensor must have the same amount of features as the training data 
        one used on the last `fit()` call, and the same batch size if this is different 
        than one.

        The different output types are: 
        - 'logits' (default): Returns the raw logits over the buckets for all the test 
          targets. Shape `(B?, test_size, n_buckets)`.
        - 'probs'/'proba'/'probabilities': Returns the probability distribution over the 
          buckets for all test targets. Shape `(B?, test_size, n_buckets)`.
        - 'values': Returns the mean, variance and standard deviation of the predicted 
          distributions for all the test targets. Shape `(B?, test_size, 3)`.
        - 'mean': Returns the mean of the predicted distributions for all the test targets. 
          Shape `(B?, test_size)`.

        For simple regression the use of 'mean' is recommended. If the model is in inference 
        mode (default), it will always output CPU tensors and squeeze them if possible.

        You can also specify the `torch.dtype` used in the forward pass if using CUDA, 
        this defaults to `torch.bfloat16`.
        '''
        # Sanity check
        if not hasattr(self, 'X_train'):
            raise RuntimeError("Please call fit() before calling predict().")

        # Preprocess tensors for encoders
        X, y, test_size = MyRegressorPFN.reshape_concatenate_pre_encoder(
            self.config.device, self.X_train, self.y_train, X_test
        )

        # If it does not require grad don't use it
        grad_context = torch.inference_mode() if not self.config.requires_grad else nullcontext()
        # If allows for mixed precisions use them
        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if (amp_dtype is not None and self.config.device == "cuda") else nullcontext()

        # Apply contexts before forward
        with grad_context:
            with amp_context:
            
                # Encode X and y
                emb_X = self.encoder_x(X)             # (B, S, Fg, emb_dim)
                emb_y = self.encoder_y(y, test_size)  # (B, S,  1, emb_dim)

                # Concatenate vectors to obtain the input tokens
                emb = torch.cat([emb_X,emb_y], dim = 2) # (B, S, Fg+1, emb_dim)

                # Add thinking rows to get the final transformer input
                trans_in = self.add_thinking_tokens(emb) # (B, S+T, Fg+1, emb_dim)

                # Send tokens through the transformer
                trans_out = self.transformer(trans_in, test_size) # (B, S+T, Fg+1, emb_dim)

                # Get logits from the decoder
                logits = self.decoder(trans_out, test_size) # (B, test_size, n_buckets)

                # Apply temperature if required
                if self.config.temperature != 1.0:
                    logits /= self.config.temperature # (B, test_size, n_buckets)

        # If not training do not return BF16 or CUDA tensors
        if not self.config.requires_grad:
            logits = logits.to(dtype=torch.float32).cpu()
            # Also squeeze the output if you can
            while logits.shape[0] == 1:
                logits = logits.squeeze(0)

        # Return corresponding output
        if output == 'logits':
            return logits # (B, test_size, n_buckets)

        probs = torch.softmax(logits, dim=-1)

        if output in ('probs','proba','probabilities'):
            return probs # (B, test_size, n_buckets)
        
        if output in ('values','mean'):
            mean, var, std = BucketOps.probs_to_mean_var_std(probs)

            if output == 'values':
                return torch.stack([mean, var, std], dim=-1) # (B, test_size, 3)
            else:
                return mean # (B, test_size)
        
        raise RuntimeError(f"Unknown output type found: {output}")
    