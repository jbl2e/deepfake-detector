# src/models.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from peft import LoraConfig, get_peft_model


def _maybe_hf_login_from_env(env_key: str = "HF_TOKEN") -> None:
    token = os.getenv(env_key)
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
        except Exception:
            pass


def _resolve_from(project_root: Path, p: str) -> Path:
    pth = Path(p).expanduser()
    if not pth.is_absolute():
        pth = project_root / pth
    return pth.resolve()


class DeepfakeDetectorBCE(nn.Module):
    """
    원본 구조 유지:
      - backbone(pixel_values) -> last_hidden_state tokens
      - reg1=tokens[:,1], reg2=tokens[:,2], patch_mean=tokens[:,patch_start:].mean(1)
      - concat -> MLP -> 1 logit (BCEWithLogitsLoss)
    """
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        reg_indices: Tuple[int, int] = (1, 2),
        patch_start_index: int = 5,
    ):
        super().__init__()
        self.backbone = backbone
        self.reg_indices = reg_indices
        self.patch_start_index = patch_start_index

        # reg1(1024)+reg2(1024)+patch_mean(1024)=3072 (DINOv3-L 기준)
        self.classifier = nn.Sequential(
            nn.Linear(3072, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state  # [B, T, C]

        reg1 = tokens[:, self.reg_indices[0], :]
        reg2 = tokens[:, self.reg_indices[1], :]

        patch_tokens = tokens[:, self.patch_start_index :, :]
        patch_mean = patch_tokens.mean(dim=1)

        feat = torch.cat([reg1, reg2, patch_mean], dim=1)  # [B, 3072]
        return self.classifier(feat)  # [B, 1]


# def build_processor(model_id: str):
#     _maybe_hf_login_from_env()
#     return AutoImageProcessor.from_pretrained(model_id)

def build_processor(cfg: Dict[str, Any], project_root: Path):
    offline_only = bool(cfg["model"].get("offline_only", False))
    local_dir = cfg["model"].get("local_backbone_dir")

    # ✅ 로컬 우선
    if local_dir:
        local_path = _resolve_from(project_root, local_dir)
        if local_path.exists():
            return AutoImageProcessor.from_pretrained(
                str(local_path),
                local_files_only=True,  # 로컬만 사용
            )

    # ✅ fallback (원하면)
    if offline_only:
        raise RuntimeError(
            "[OFFLINE] local_backbone_dir not found, and offline_only=true.\n"
            "Please place backbone files under model/dinov3_backbone."
        )

    _maybe_hf_login_from_env()
    return AutoImageProcessor.from_pretrained(cfg["model"]["hf_model_id"])

# def build_backbone_with_lora(model_id: str, lora_cfg: Dict[str, Any]):
#     _maybe_hf_login_from_env()
#     backbone = AutoModel.from_pretrained(model_id)

#     # 원본과 동일: 기본 가중치 Frozen :contentReference[oaicite:2]{index=2}
#     for p in backbone.parameters():
#         p.requires_grad = False

#     # 원본 LoRA 설정 유지 :contentReference[oaicite:3]{index=3}
#     peft_cfg = LoraConfig(
#         r=int(lora_cfg["r"]),
#         lora_alpha=int(lora_cfg["alpha"]),
#         target_modules=list(lora_cfg["target_modules"]),
#         lora_dropout=float(lora_cfg["dropout"]),
#         bias="none",
#     )
#     return get_peft_model(backbone, peft_cfg)

def build_backbone_with_lora(cfg: Dict[str, Any], project_root: Path):
    offline_only = bool(cfg["model"].get("offline_only", False))
    local_dir = cfg["model"].get("local_backbone_dir")

    # ✅ 로컬 우선
    if local_dir:
        local_path = _resolve_from(project_root, local_dir)
        if local_path.exists():
            backbone = AutoModel.from_pretrained(
                str(local_path),
                local_files_only=True,
            )
        else:
            if offline_only:
                raise RuntimeError(
                    "[OFFLINE] local_backbone_dir not found, and offline_only=true.\n"
                    f"missing: {local_path}"
                )
            _maybe_hf_login_from_env()
            backbone = AutoModel.from_pretrained(cfg["model"]["hf_model_id"])
    else:
        if offline_only:
            raise RuntimeError("[OFFLINE] local_backbone_dir is missing in config.")
        _maybe_hf_login_from_env()
        backbone = AutoModel.from_pretrained(cfg["model"]["hf_model_id"])

    # ✅ backbone freeze
    for p in backbone.parameters():
        p.requires_grad = False

    lora_cfg = cfg["model"]["lora"]
    peft_cfg = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
    )
    return get_peft_model(backbone, peft_cfg)

def build_detector(cfg: Dict[str, Any], device: torch.device, project_root: Path):
    processor = build_processor(cfg, project_root)
    backbone = build_backbone_with_lora(cfg, project_root)

    detector = DeepfakeDetectorBCE(
        backbone=backbone,
        hidden_dim=int(cfg["model"]["head"]["hidden_dim"]),
        dropout=float(cfg["model"]["head"]["dropout"]),
        reg_indices=tuple(cfg["model"]["feature"]["reg_indices"]),
        patch_start_index=int(cfg["model"]["feature"]["patch_start_index"]),
    ).to(device)

    return detector, processor


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable}
