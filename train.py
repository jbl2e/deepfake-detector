# train.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import shutil
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
import yaml

from src.models import build_detector, count_trainable_parameters
from src.dataset import LoaderConfig, create_loaders
from src.utils import get_transforms, extract_zip

import os
import zipfile
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_from(base_dir: Path, p: str) -> Path:
    pth = Path(p).expanduser()
    if not pth.is_absolute():
        pth = (base_dir / pth)
    return pth.resolve()

def _dir_has_any_files(p: Path) -> bool:
    return p.exists() and any(x.is_file() for x in p.rglob("*"))

def _flatten_nested_same_dir(dest_dir: Path) -> None:
    """
    dest_dir 안이 dest_dir/train_full_data/... 가 아니라
    dest_dir/train_full_data/train_full_data/... 로 2겹이면 1겹만 끌어올림.

    즉 dest_dir.name == "train_full_data" 일 때:
      dest_dir / "train_full_data" 가 유일한 1개 디렉토리이면 flatten
    """
    dest_dir = Path(dest_dir)
    inner = dest_dir / dest_dir.name  # dest_dir/train_full_data
    if not inner.exists() or not inner.is_dir():
        return

    # dest_dir 내부가 inner 하나만 있을 때만 flatten (안전)
    items = list(dest_dir.iterdir())
    if len(items) != 1 or items[0].name != dest_dir.name:
        return

    tmp = dest_dir.parent / f"{dest_dir.name}__tmp_flatten"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    for child in inner.iterdir():
        shutil.move(str(child), str(tmp / child.name))

    shutil.rmtree(inner, ignore_errors=True)

    for child in tmp.iterdir():
        shutil.move(str(child), str(dest_dir / child.name))

    shutil.rmtree(tmp, ignore_errors=True)

def maybe_extract_archives(cfg: Dict[str, Any], project_root: Path, runtime_root: Path) -> None:
    """
    ✅ config.data.archives[*].dest 를 그대로 존중해서 그 위치로 extract
    ✅ marker도 dest 아래에 생성
    ✅ dest가 train_full_data인 경우, dest/train_full_data 2겹만 1번 flatten
    """
    archives = cfg.get("data", {}).get("archives", [])
    if not archives:
        return

    for item in archives:
        archive_path = resolve_from(project_root, item["path"])   # ✅ Drive에서 zip 찾기
        dest_dir     = resolve_from(runtime_root, item["dest"])   # ✅ /content에 풀기
        marker_name = str(item.get("marker", "")).strip()
        marker_path = (dest_dir / marker_name) if marker_name else None

        if not archive_path.exists():
            print(f"[WARN] archive not found, skip: {archive_path}")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)

        # marker가 있고, dest에 파일이 있으면 스킵
        if marker_path is not None and marker_path.exists() and _dir_has_any_files(dest_dir):
            print(f"[Extract] marker exists -> skip: {marker_path}")
            continue

        # marker는 있는데 dest가 비었으면(깨짐) marker 제거 후 재추출
        if marker_path is not None and marker_path.exists() and not _dir_has_any_files(dest_dir):
            print(f"[WARN] marker exists but dest empty -> re-extract: {dest_dir}")
            try:
                marker_path.unlink()
            except Exception:
                pass

        print(f"[Extract] zip: {archive_path.name} -> {dest_dir}")
        extract_zip(archive_path, dest_dir)

        # dest가 train_full_data라면 2겹만 정리
        _flatten_nested_same_dir(dest_dir)

        # 최종 검증
        if not _dir_has_any_files(dest_dir):
            raise RuntimeError(
                f"[Extract-ERROR] dest empty after extract.\n"
                f"dest: {dest_dir}\n"
                f"zip:  {archive_path}"
            )

        if marker_path is not None:
            marker_path.write_text("ok", encoding="utf-8")

        print(f"[Extract] done -> {dest_dir}")


def _list_images(p: Path) -> List[str]:
    if not p.exists():
        return []
    exts = IMG_EXTS
    return [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts]

def report_sources_counts(sources: Dict[str, List[str]]) -> None:
    print("=" * 70)
    print("[Dataset Source Scan]")
    print("=" * 70)
    for label_name in ["real", "fake"]:
        total = 0
        print(f"\n[{label_name.upper()}]")
        for path_str in sources.get(label_name, []):
            p = Path(path_str)
            files = _list_images(p)
            total += len(files)
            status = f"{len(files):,}" if p.exists() else "MISSING"
            print(f" - {str(p)}  ->  {status}")
        print(f" => {label_name} TOTAL: {total:,}")
    print("=" * 70)

def get_smart_balanced_datasets_v5(
    sources: Dict[str, List[str]],
    train_target=(164000, 175000),
    valid_target=(35000, 35000),
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:

    total_real_target = int(train_target[0]) + int(valid_target[0])
    total_fake_target = int(train_target[1]) + int(valid_target[1])

    real_train_ratio = int(train_target[0]) / max(1, total_real_target)
    fake_train_ratio = int(train_target[1]) / max(1, total_fake_target)

    def process_label(paths, total_target, train_ratio, label_name):
        train_pool = []
        valid_pool = []
        large_sources = []

        print(f"\n🔍 {label_name} processing (target={total_target:,}, train_ratio={train_ratio:.1%})")

        current_total = 0
        active_paths = [p for p in paths if Path(p).exists()]

        for path_str in active_paths:
            p = Path(path_str)
            files = _list_images(p)
            files = list(set(files))
            random.shuffle(files)

            split_idx = int(len(files) * train_ratio)
            f_train = files[:split_idx]
            f_valid = files[split_idx:]

            # 너 로직 유지: refined_dataset or small source는 비율 분할로 통째 반영
            if ("refined_dataset" in path_str) or (len(files) < 9000):
                train_pool.extend(f_train)
                valid_pool.extend(f_valid)
                current_total += len(files)
                print(f"   ✅ [ratio split] {path_str:<55} | Tr {len(f_train):>7,} / Vl {len(f_valid):>7,}")
            else:
                large_sources.append({"path": path_str, "train": f_train, "valid": f_valid})

        remaining_needed = int(total_target) - int(current_total)

        if remaining_needed > 0 and large_sources:
            for i, src in enumerate(large_sources):
                denom = max(1, (len(large_sources) - i))
                num_to_take = remaining_needed // denom

                files_total = src["train"] + src["valid"]
                actual_take = min(len(files_total), num_to_take)

                take_idx = int(actual_take * train_ratio)

                selected_train = src["train"][:take_idx]
                selected_valid = src["valid"][:(actual_take - take_idx)]

                train_pool.extend(selected_train)
                valid_pool.extend(selected_valid)

                remaining_needed -= (len(selected_train) + len(selected_valid))
                print(f"   ⚖️ [top-up]     {src['path']:<55} | Tr {len(selected_train):>7,} / Vl {len(selected_valid):>7,}")

        # ✅ 최종 목표보다 더 많아질 수 있으니 타겟 컷(안전)
        # (여기서 overshoot 방지. 너 목표 수량을 정확히 맞추려면 필요)
        random.shuffle(train_pool)
        random.shuffle(valid_pool)

        need_train = int(total_target * train_ratio)
        need_valid = int(total_target - need_train)

        train_pool = train_pool[:need_train]
        valid_pool = valid_pool[:need_valid]

        return train_pool, valid_pool

    real_train, real_valid = process_label(sources["real"], total_real_target, real_train_ratio, "REAL")
    fake_train, fake_valid = process_label(sources["fake"], total_fake_target, fake_train_ratio, "FAKE")

    train_data = [(f, 0) for f in real_train] + [(f, 1) for f in fake_train]
    valid_data = [(f, 0) for f in real_valid] + [(f, 1) for f in fake_valid]

    random.shuffle(train_data)
    random.shuffle(valid_data)

    return train_data, valid_data



def build_datalists(cfg: Dict[str, Any], cfg_dir: Path):
    data_cfg = cfg.get("data", {})
    sources_cfg = data_cfg.get("sources")
    if sources_cfg is None:
        raise RuntimeError("[CONFIG-ERROR] data.sources missing in config.yaml")

    sources = {
        "real": [str(resolve_from(cfg_dir, s)) for s in sources_cfg.get("real", [])],
        "fake": [str(resolve_from(cfg_dir, s)) for s in sources_cfg.get("fake", [])],
    }

    report_sources_counts(sources)

    split_targets = data_cfg.get("split_targets", {})
    train_target = tuple(split_targets.get("train_target", [164000, 175000]))
    valid_target = tuple(split_targets.get("valid_target", [35000, 35000]))

    return get_smart_balanced_datasets_v5(
        sources=sources,
        train_target=train_target,
        valid_target=valid_target,
    )




# -------------------------
# Epoch runner (그대로 유지)
# -------------------------
def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    phase: str,
    accumulation_steps: int = 1,
):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_labels = []
    all_probs = []

    total = len(loader)
    pbar = tqdm(loader, total=total, desc=f"{phase}", leave=False)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        with torch.set_grad_enabled(is_train):
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            if is_train:
                loss_scaled = loss / accumulation_steps
                scaler.scale(loss_scaled).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * images.size(0)

        if batch_idx % 20 == 0:
            pbar.set_postfix(loss=float(loss.item()))

        probs = torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().float().cpu().numpy().reshape(-1).tolist())

    epoch_loss = running_loss / max(1, len(loader.dataset))
    try:
        epoch_auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
    except Exception:
        epoch_auc = float("nan")

    return epoch_loss, epoch_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = Path(args.config).expanduser().resolve()
    cfg_dir  = cfg_path.parent
    project_root = cfg_dir.parent   

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["project"]["seed"]))
    # ✅ Colab 로컬 런타임(빠른 디스크) 루트
    #    기본값: /content/dacon_runtime
    runtime_root = Path(os.environ.get("RUNTIME_ROOT", "/content")).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    print("[RUNTIME] runtime_root:", runtime_root)

    print("[CFG] cfg_path:", cfg_path)
    print("[CFG] cfg_dir :", cfg_dir)
    print("[CFG] cwd     :", Path.cwd().resolve())

    # ✅ zip extract
    maybe_extract_archives(cfg, project_root, runtime_root)
    # extraction 후
    print("[Check] extracted root exists:", (resolve_from(runtime_root, "./train_data/train_full_data")).exists())

    # ✅ output dirs
    out_dir = resolve_from(project_root, cfg["paths"]["output_dir"])
    ckpt_dir = resolve_from(project_root, cfg["paths"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # model, processor = build_detector(cfg, device=device)
   
    model, processor = build_detector(cfg, device=device, project_root=project_root)
    stats = count_trainable_parameters(model)
    print(f"[Model] total={stats['total']:,} trainable={stats['trainable']:,}")

    train_data, valid_data = build_datalists(cfg, runtime_root)

    tr = Counter(y for _, y in train_data)
    vl = Counter(y for _, y in valid_data)
    print(f"[Split] TRAIN: real={tr.get(0,0)} fake={tr.get(1,0)} total={len(train_data)}")
    print(f"[Split] VALID: real={vl.get(0,0)} fake={vl.get(1,0)} total={len(valid_data)}")

    if len(train_data) == 0 or len(valid_data) == 0:
        raise RuntimeError(
            f"Empty dataset! train={len(train_data)}, valid={len(valid_data)}\n"
            f"Check extraction paths and that each split folder contains Real/ and Fake/ with images."
        )

    # 4) transforms (그대로)
    input_size = int(cfg["train"]["input_size"])
    train_tf, valid_tf = get_transforms(
        input_size=input_size,
        mean=tuple(processor.image_mean),
        std=tuple(processor.image_std),
    )

    # 5) loaders (그대로)
    loader_cfg = LoaderConfig(
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=bool(cfg["train"].get("pin_memory", True)),
        drop_last=bool(cfg["train"].get("drop_last", True)),
    )

    train_loader, valid_loader = create_loaders(
        train_data=train_data,
        valid_data=valid_data,
        train_transform=train_tf,
        valid_transform=valid_tf,
        cfg=loader_cfg,
        use_weighted_sampler=bool(cfg["train"].get("use_weighted_sampler", True)),
    )

    # 6) loss / optimizer (그대로)
    criterion = torch.nn.BCEWithLogitsLoss()
    opt_cfg = cfg["train"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))

    epochs = int(cfg["train"]["epochs"])
    accumulation_steps = int(cfg["train"]["accumulation_steps"])

    best_auc = -1.0
    best_path = out_dir / cfg["train"]["save"]["best_name"]

    for epoch in range(1, epochs + 1):
        tr_loss, tr_auc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            phase="train", accumulation_steps=accumulation_steps
        )
        vl_loss, vl_auc = run_epoch(
            model, valid_loader, criterion, optimizer, scaler, device,
            phase="valid", accumulation_steps=1
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train loss={tr_loss:.4f} auc={tr_auc:.4f} | "
            f"valid loss={vl_loss:.4f} auc={vl_auc:.4f}"
        )

        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "cfg": cfg}, ckpt_path)
        if (not ckpt_path.exists()) or (ckpt_path.stat().st_size < 1024 * 1024):
          raise RuntimeError(f"[SAVE-ERROR] ckpt not created or too small: {ckpt_path}")


        if (not np.isnan(vl_auc)) and (vl_auc > best_auc):
            best_auc = vl_auc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "cfg": cfg}, best_path)
            print(f"  ✅ best updated: auc={best_auc:.4f} -> {best_path}")
            if (not best_path.exists()) or (best_path.stat().st_size < 1024 * 1024):
              raise RuntimeError(f"[SAVE-ERROR] best_path not created or too small: {best_path}")

    print(f"[Done] best_auc={best_auc:.4f} saved={best_path}")


if __name__ == "__main__":
    main()
