from libero.lifelong.models import *
from models import *


class LoraBCTPolicy(BCTransformerPolicy):
    def __init__(self, cfg, shape_meta):
        super(LoraBCTPolicy, self).__init__(cfg, shape_meta)

        policy_cfg = cfg.policy
        # introduce the lora adapters for spatial and temporal encoders (maybe also the policy head)
        self.temporal_transformer = LoraTransformerDecoder(
            input_size=policy_cfg.embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
            lora_rank=cfg.adaptation.lora_rank,
        )
