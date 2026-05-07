import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phase1.utils import FeatureWriter, read_manifest, load_local_checkpoint, pad_batch
from evo.tokenizer import CharLevelTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract layer-wise mean-pooled features from Evo.")
    parser.add_argument("--manifest", default="data/phase1/manifest.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase1/features")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    sequences = [r.sequence for r in records]
    labels = np.array([r.label for r in records], dtype=np.int64)
    splits = np.array([r.split for r in records])
    record_ids = np.array([r.record_id for r in records])

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)
    np.save(os.path.join(args.out_dir, "splits.npy"), splits)
    np.save(os.path.join(args.out_dir, "ids.npy"), record_ids)

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    model.eval()

    tokenizer = CharLevelTokenizer(512)

    num_layers = len(model.blocks)
    writer = FeatureWriter(args.out_dir, num_layers=num_layers, chunk_size=args.chunk_size)

    state: Dict[str, torch.Tensor] = {"mask": None}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            mask = state["mask"]
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            writer.add(layer_idx, pooled.detach().float().cpu())
        return hook

    hooks = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(model.blocks)]

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), args.batch_size), desc="Extracting"):
            batch = sequences[start : start + args.batch_size]
            batch = [seq[: args.max_length] for seq in batch]
            token_ids = tokenizer.tokenize_batch(batch)
            input_ids, mask = pad_batch(token_ids, tokenizer.pad_id)
            input_ids = input_ids.to(args.device)
            state["mask"] = mask.to(args.device)
            _ = model(input_ids, padding_mask=state["mask"])

    writer.flush_all()
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
