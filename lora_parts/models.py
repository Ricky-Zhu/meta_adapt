from libero.lifelong.models.bc_transformer_policy import *
import torch
from libero.lifelong.models.modules.transformer_modules import *
import loralib as lora


class LoraAttention(nn.Module):
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0, lora_rank=10):
        super().__init__()

        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )
        self.q_lora = lora.Linear(dim, num_heads * head_output_size, r=lora_rank)
        self.v_lora = lora.Linear(dim, num_heads * head_output_size, r=lora_rank)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (qkv[0], qkv[1], qkv[2])
        q_lora = self.q_lora(x).reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)[0]
        v_lora = self.v_lora(x).reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)[0]
        # manipulate q and v
        q = q + q_lora
        v = v + v_lora

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.output_layer(out)


class LoraTransformerFeedForwardNN(TransformerFeedForwardNN):
    def __init__(self, dim, hidden_dim, dropout, lora_rank):
        super(LoraTransformerFeedForwardNN, self).__init__(dim, hidden_dim, dropout)
        self.lora_1 = lora.Linear(dim, hidden_dim, r=lora_rank)
        self.lora_2 = lora.Linear(hidden_dim, dim, r=lora_rank)

    def forward(self, x):
        x_lora1 = self.lora_1(x)
        x = self.l1(x)
        x = x + x_lora1
        x = self.gelu1(x)
        x = self.dp1(x)
        x_lora2 = self.lora_2(x)
        x = self.l2(x)
        x = x + x_lora2
        x = self.dp2(x)
        return x


class LoraTransformerDecoder(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            num_heads,
            head_output_size,
            mlp_hidden_size,
            dropout,
            **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_rank = kwargs['lora_rank']
        self.attention_output = {}

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        LoraAttention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                            lora_rank=self.lora_rank,
                        ),
                        Norm(input_size),
                        LoraTransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout, lora_rank=self.lora_rank
                        ),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
                (self.num_elements is None)
                or (self.seq_len is None)
                or (self.num_elements != input_shape[2])
                or (self.seq_len != input_shape[1])
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

    def forward(self, x, mask=None):
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask))
            elif self.mask is not None:
                x = x + drop_path(att(att_norm(x), self.mask))
            else:  # no masking, just use full attention
                x = x + drop_path(att(att_norm(x)))

            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    model = LoraAttention(dim=128)
    X = torch.randn(8, 10, 128)
    y = model(X)
    print('sd')
