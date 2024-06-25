from libero.lifelong.models import *
from .lora_models import LoraTransformerDecoder


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

class LoraBCViLTPolicy(BCViLTPolicy):
    def __init__(self, cfg, shape_meta):
        super(LoraBCViLTPolicy, self).__init__(cfg, shape_meta)

        policy_cfg = cfg.policy
        # introduce the lora adapters for spatial and temporal encoders (maybe also the policy head)

        self.spatial_transformer = LoraTransformerDecoder(
            input_size=policy_cfg.embed_size,
            num_layers=policy_cfg.spatial_transformer_num_layers,
            num_heads=policy_cfg.spatial_transformer_num_heads,
            head_output_size=policy_cfg.spatial_transformer_head_output_size,
            mlp_hidden_size=policy_cfg.spatial_transformer_mlp_hidden_size,
            dropout=policy_cfg.spatial_transformer_dropout,
            lora_rank=cfg.adaptation.lora_rank,
        )

        if policy_cfg.spatial_down_sample:
            temporal_embed_size = policy_cfg.spatial_down_sample_embed_size
        else:
            temporal_embed_size = policy_cfg.embed_size

        self.temporal_transformer = LoraTransformerDecoder(
            input_size=temporal_embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
            lora_rank=cfg.adaptation.lora_rank,
        )