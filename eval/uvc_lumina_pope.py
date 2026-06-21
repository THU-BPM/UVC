#!/usr/bin/env python3
"""
POPE evaluation with UVC intervention for Lumina-DiMOO.
  - Head selection: **Probe AUC** (LogisticRegression, Clean vs Global / Clean vs Instance)
  - Intervention direction: **CoM** (pos_mean - neg_mean, unit-normalised, proj_std scaled)
  - Injection hook: **pre-hook on attn_out INPUT** (per-head concat before linear mixing)
  - Mask scope: **gen-region mask** (only MASK tokens within last gen_len positions)
  - 32 heads, head_dim=128
"""
import torch
import json
from einops import rearrange
import numpy as np
import os
from tqdm import tqdm
import argparse
import sys
import re
import random
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from concurrent.futures import ThreadPoolExecutor, as_completed

# path setup
_script_dir = os.path.dirname(os.path.abspath(__file__))
_dict_dir   = os.path.dirname(_script_dir)
_work_dir   = os.path.dirname(_dict_dir)

_lumina_candidates = [
    os.environ.get("LUMINA_ROOT", ""),
    os.path.join(_work_dir, "Lumina-DiMOO"),
]
for _p in _lumina_candidates:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.image_utils import (
    encode_img_with_breaks, calculate_vq_params,
    generate_crop_size_list, var_center_crop, add_break_line,
)
from utils.prompt_utils import generate_multimodal_understanding_prompt
from generators.text_understanding_generator import generate_text_understanding
from transformers import AutoTokenizer, set_seed
from diffusers import VQModel
from PIL import Image

NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIM = 128
MASK_ID  = SPECIAL_TOKENS["mask_token"]    # 126336
NEW_LINE = SPECIAL_TOKENS["newline_token"]
BOA      = SPECIAL_TOKENS["answer_start"]  # 126354
EOA      = SPECIAL_TOKENS["answer_end"]    # 126355


# Probe training & head selection
def flattened_idx_to_layer_head(idx, num_heads):
    return idx // num_heads, idx % num_heads


def load_vec_as_B_L_H_D(path, num_heads=NUM_HEADS):
    """Load vectors and handle various shapes (3D/4D)."""
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 4:
        arr = arr.mean(axis=2)
    if arr.ndim == 3:
        return rearrange(arr, "b l (h d) -> b l h d", h=num_heads)
    if arr.ndim == 2:
        return rearrange(arr, "l (h d) -> 1 l h d", h=num_heads)
    raise ValueError(f"Unsupported vector shape: {arr.shape} for {path}")


def train_probe(layer, head, X, X_labels, kf):
    X_layer = np.array(X[:, layer, head, :])
    fold_aucs = []
    for train_idx, test_idx in kf.split(X_layer):
        probe = LogisticRegression(solver='saga', max_iter=1000, n_jobs=32)
        probe.fit(X_layer[train_idx], X_labels[train_idx])
        test_labels = X_labels[test_idx]
        if len(np.unique(test_labels)) < 2:
            fold_aucs.append(0.5)
        else:
            scores = probe.predict_proba(X_layer[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(test_labels, scores))
    return (layer, head, float(np.mean(fold_aucs)), probe)


def build_interventions_from_probes(
    clean_path, contrast_path, num_heads_k,
    label="global", length_clip=1500,
):
    """
    Train probes (Clean vs Contrast), select top-K heads by AUC,
    compute CoM directions and proj_std for each selected head.
    Returns: interventions dict, top_heads list, AUC score array
    """
    vis = load_vec_as_B_L_H_D(clean_path)
    con = load_vec_as_B_L_H_D(contrast_path)

    B = min(vis.shape[0], con.shape[0])
    if length_clip and length_clip > 0:
        B = min(B, int(length_clip))
    vis, con = vis[:B], con[:B]

    # CoM directions (for injection): clean_mean - contrast_mean
    com_directions = []
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            vm = vis[:, layer, head, :].mean(axis=0)
            cm = con[:, layer, head, :].mean(axis=0)
            com_directions.append(vm - cm)

    # Train probes for head selection
    X = np.concatenate((vis, con), axis=0)
    n = vis.shape[0]
    labels = np.zeros(n * 2)
    labels[n:] = 1
    indices = np.arange(n * 2)
    np.random.shuffle(indices)
    X = X[indices]
    labels = labels[indices]

    kf = KFold(n_splits=2)
    accuracies = np.empty((NUM_LAYERS, NUM_HEADS), dtype=float)
    probes = {}

    print(f"  Training {label} probes (B={B}) ...")
    with ThreadPoolExecutor(max_workers=64) as exe:
        futs = []
        for l in range(NUM_LAYERS):
            for h in range(NUM_HEADS):
                futs.append(exe.submit(train_probe, l, h, X, labels, kf))
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"  {label} probes"):
            l, h, acc, p = fut.result()
            probes[(l, h)] = p
            accuracies[l, h] = acc

    top_flat = np.argsort(accuracies.flatten())[::-1][:num_heads_k]
    top_heads = [flattened_idx_to_layer_head(idx, NUM_HEADS) for idx in top_flat]

    print(f"  {label} top-10 heads: {top_heads[:10]}")
    print(f"  {label} AUC range: [{accuracies.min():.4f}, {accuracies.max():.4f}]")

    # Build interventions dict with CoM direction + proj_std
    interventions = {}
    for layer, head in top_heads:
        key = f"model.transformer.blocks.{layer}.attn_out"
        interventions.setdefault(key, [])
    for layer, head in top_heads:
        direction = com_directions[layer * NUM_HEADS + head]
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        activations = X[:, layer, head, :]
        proj_vals = activations @ direction.T
        proj_val_std = float(np.std(proj_vals))
        key = f"model.transformer.blocks.{layer}.attn_out"
        interventions[key].append((head, direction.squeeze().astype(np.float32), proj_val_std))
    for key in interventions:
        interventions[key] = sorted(interventions[key], key=lambda x: x[0])

    return interventions, top_heads, accuracies


# Hook registration - gen-region mask injection
def register_hooks(model, interventions_global, interventions_instance,
                   alpha, beta, gen_len, intervention_type):
    """
    Register PRE-HOOKS on attn_out INPUT (per-head concat before linear mixing).
    Mask scope: only MASK tokens within the last gen_len positions.
    """
    # Wrap model.forward to track mask positions
    if not hasattr(model, "_uvc_mask_index_wrapped"):
        model._uvc_mask_index = None
        _orig_forward = model.forward

        def _forward_with_mask_tracking(input_ids=None, **kwargs):
            if input_ids is not None:
                try:
                    model._uvc_mask_index = (input_ids == MASK_ID)
                except Exception:
                    model._uvc_mask_index = None
            else:
                model._uvc_mask_index = None
            return _orig_forward(input_ids=input_ids, **kwargs)

        model.forward = _forward_with_mask_tracking
        model._uvc_mask_index_wrapped = True

    # Scale for both type
    eff_alpha = alpha * 0.5 if intervention_type == "both" else alpha
    eff_beta = beta * 0.5 if intervention_type == "both" else beta

    all_keys = set()
    if interventions_global:
        all_keys |= set(interventions_global.keys())
    if interventions_instance:
        all_keys |= set(interventions_instance.keys())

    modules = dict(model.named_modules())
    handles = []
    _first_call = [True]

    def hook_factory(layer_key):
        def hook_fn(module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return inputs
            hidden = inputs[0]
            head_out = rearrange(hidden, "b s (h d) -> b s h d", h=NUM_HEADS)
            seq_len = int(head_out.shape[1])
            gen_start = max(0, seq_len - gen_len)

            # --- mask tracking: gen-region mask only ---
            mask_index = getattr(model, "_uvc_mask_index", None)
            tail_len = 0
            if mask_index is not None and isinstance(mask_index, torch.Tensor) and mask_index.ndim == 2:
                tail_len = int(min(gen_len, seq_len, int(mask_index.shape[1])))
            tail_start = int(seq_len - tail_len) if tail_len > 0 else gen_start

            if _first_call[0]:
                mask_cnt = int(mask_index.sum().item()) if mask_index is not None else -1
                print(f"[HOOK] {layer_key} seq={seq_len} gen_start={gen_start} "
                      f"tail_len={tail_len} tail_start={tail_start} mask_cnt={mask_cnt}")
                _first_call[0] = False

            # Image-level intervention
            if interventions_global and layer_key in interventions_global:
                for head, d_unit, proj_std in interventions_global[layer_key]:
                    direction = torch.tensor(d_unit, dtype=head_out.dtype, device=head_out.device)
                    intervention = eff_alpha * proj_std * direction
                    if tail_len > 0 and mask_index is not None:
                        mask_tail = mask_index[:, -tail_len:].to(
                            device=head_out.device, dtype=torch.bool).unsqueeze(-1)
                        head_out[:, tail_start:, head, :] = torch.where(
                            mask_tail,
                            head_out[:, tail_start:, head, :] + intervention,
                            head_out[:, tail_start:, head, :],
                        )
                    else:
                        head_out[:, gen_start:, head, :] += intervention

            # Instance-scale intervention
            if interventions_instance and layer_key in interventions_instance:
                for head, d_unit, proj_std in interventions_instance[layer_key]:
                    direction = torch.tensor(d_unit, dtype=head_out.dtype, device=head_out.device)
                    intervention = eff_beta * proj_std * direction
                    if tail_len > 0 and mask_index is not None:
                        mask_tail = mask_index[:, -tail_len:].to(
                            device=head_out.device, dtype=torch.bool).unsqueeze(-1)
                        head_out[:, tail_start:, head, :] = torch.where(
                            mask_tail,
                            head_out[:, tail_start:, head, :] + intervention,
                            head_out[:, tail_start:, head, :],
                        )
                    else:
                        head_out[:, gen_start:, head, :] += intervention

            modified = rearrange(head_out, "b s h d -> b s (h d)")
            return (modified,) + inputs[1:]
        return hook_fn

    for name, module in modules.items():
        if not name.endswith(".attn_out"):
            continue
        if name in all_keys:
            handles.append(module.register_forward_pre_hook(hook_factory(name)))

    print(f"[UVC] Registered {len(handles)} hooks | type={intervention_type} "
          f"alpha={eff_alpha:.1f} beta={eff_beta:.1f} gen_len={gen_len}")
    return handles


def parse_yes_no_from_text(text):
    if not text:
        return "unknown"
    text = str(text).strip()
    if not text:
        return "unknown"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        first_tok = line.split(maxsplit=1)[0].strip()
        first_tok = re.sub(r"^[^A-Za-z]+", "", first_tok)
        first_l = first_tok.lower()
        if first_l.startswith("yes"):
            return "yes"
        if first_l.startswith("no"):
            return "no"
        return "unknown"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pope_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--vae_ckpt", type=str, default=None)

    ap.add_argument("--clean_vector", type=str, required=True, help="Clean-reference vectors")
    ap.add_argument("--global_vector", type=str, default=None, help="Global-scale degraded vectors")
    ap.add_argument("--instance_vector", type=str, default=None, help="Instance-scale degraded vectors")

    ap.add_argument("--type", type=str, default="global", choices=["global", "instance", "both"])
    ap.add_argument("--num_heads", type=int, default=32, help="K: number of top heads")
    ap.add_argument("--alpha", type=float, default=20, help="Global-scale intervention strength")
    ap.add_argument("--beta", type=float, default=None, help="Instance-scale strength (default=alpha)")

    ap.add_argument("--num_samples", type=int, default=400)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument("--remasking", type=str, default="low_confidence")
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--output", type=str, required=True)

    ap.add_argument("--probe_file", type=str, default=None, help="Save/load probe data")
    ap.add_argument("--train_probe", action="store_true", help="Force retrain")
    args = ap.parse_args()

    if args.beta is None:
        args.beta = args.alpha
    if args.type in ["global", "both"] and (not args.global_vector or not os.path.exists(args.global_vector)):
        raise ValueError("--global_vector required for global/both type")
    if args.type in ["instance", "both"] and (not args.instance_vector or not os.path.exists(args.instance_vector)):
        raise ValueError("--instance_vector required for instance/both type")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1. Build interventions (probe selection + CoM direction)
    interventions_global = {}
    interventions_instance = {}

    probe_loaded = False
    if args.probe_file and os.path.exists(args.probe_file) and not args.train_probe:
        print(f"Loading probes from {args.probe_file} ...")
        with open(args.probe_file, 'rb') as f:
            pd = pickle.load(f)
        interventions_global = pd.get('interventions_global', pd.get('interventions', {}))
        interventions_instance = pd.get('interventions_instance', pd.get('interventions_object', {}))
        top_heads_global = pd.get('top_heads_global', pd.get('top_heads', []))
        top_heads_instance = pd.get('top_heads_instance', pd.get('top_heads_object', []))
        print(f"  Loaded global heads: {len(top_heads_global)}, instance heads: {len(top_heads_instance)}")
        probe_loaded = True

    if not probe_loaded:
        print("Building interventions (probe selection + CoM direction) ...")
        if args.type in ["global", "both"]:
            interventions_global, top_heads_global, _ = build_interventions_from_probes(
                args.clean_vector, args.global_vector,
                args.num_heads, label="global")

        if args.type in ["instance", "both"]:
            interventions_instance, top_heads_instance, _ = build_interventions_from_probes(
                args.clean_vector, args.instance_vector,
                args.num_heads, label="instance")

        if args.type == "instance":
            top_heads_global = []

        if args.probe_file:
            os.makedirs(os.path.dirname(args.probe_file), exist_ok=True)
            with open(args.probe_file, 'wb') as f:
                pickle.dump({
                    'interventions_global': interventions_global,
                    'interventions_instance': interventions_instance,
                    'top_heads_global': top_heads_global if args.type != "instance" else [],
                    'top_heads_instance': top_heads_instance if args.type != "global" else [],
                    'num_heads_k': args.num_heads,
                    'type': args.type,
                }, f)
            print(f"  Probes saved to {args.probe_file}")

    # 2. Load model
    print("Loading Lumina-DiMOO model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    vae_ckpt = args.vae_ckpt or args.model_path
    vqvae = VQModel.from_pretrained(vae_ckpt, subfolder="vqvae").to(device)
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)

    # 3. Register hooks
    gen_len = max(1, int(args.gen_length))
    hook_handles = register_hooks(
        model,
        interventions_global if args.type in ["global", "both"] else None,
        interventions_instance if args.type in ["instance", "both"] else None,
        alpha=args.alpha, beta=args.beta,
        gen_len=gen_len, intervention_type=args.type,
    )

    # 4. Run POPE evaluation
    print(f"Loading POPE data from {args.pope_file} ...")
    with open(args.pope_file, "r") as f:
        entries = [json.loads(line) for line in f]
    if args.num_samples:
        entries = entries[:args.num_samples]

    tp = tn = fp = fn = unk = 0
    yes_cnt = 0
    total = len(entries)
    details = []

    output_jsonl = args.output.replace('.json', '.jsonl') if args.output.endswith('.json') else args.output + '.jsonl'
    ans_file = open(output_jsonl, "w")

    base_seed = args.seed
    for item in tqdm(entries, desc="POPE Eval"):
        img_file = item["image"]
        question = item["text"]
        label = item.get("label", "").lower().strip()
        qid = item.get("question_id")

        image_path = os.path.join(args.image_folder, img_file)
        if not os.path.exists(image_path):
            unk += 1
            continue

        try:
            # Image processing (Lumina pipeline)
            raw_image = Image.open(image_path).convert("RGB")
            crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)
            image = var_center_crop(raw_image, crop_size_list=crop_size_list)
            image_w, image_h = image.size
            _, _, tg_h, tg_w = calculate_vq_params(image_h, image_w, vae_scale)
            input_img_token = encode_img_with_breaks(image, vqvae=vqvae)
            img_token = add_break_line(input_img_token, tg_h, tg_w, new_number=NEW_LINE)

            # Prompt construction
            input_prompt = generate_multimodal_understanding_prompt(question)
            input_ids = tokenizer(input_prompt)["input_ids"]
            input_token = input_ids[:-1] + img_token + input_ids[-1:]
            code_start = len(input_token) + 1  # +1 for BOA
            input_token = input_token + [BOA] + args.gen_length * [MASK_ID] + [EOA]
            input_ids_tensor = torch.tensor(input_token, device=device).unsqueeze(0)

            # Per-sample RNG
            try:
                qseed = base_seed + int(qid or 0)
            except Exception:
                qseed = base_seed
            random.seed(int(qseed))
            np.random.seed(int(qseed) % (2**32 - 1))
            torch.manual_seed(int(qseed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(qseed))

            with torch.no_grad():
                out_new = generate_text_understanding(
                    model, input_ids_tensor,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    code_start=code_start,
                )

            response = tokenizer.batch_decode(
                out_new[:, code_start:-1], skip_special_tokens=True
            )[0].strip()

        except Exception as e:
            unk += 1
            details.append({"question_id": qid, "pred": "error", "gt": label, "error": str(e)})
            continue

        pred = parse_yes_no_from_text(response)
        if pred == "yes": yes_cnt += 1
        if pred == "unknown": unk += 1

        if label == "yes":
            if pred == "yes": tp += 1
            elif pred == "no": fn += 1
        elif label == "no":
            if pred == "no": tn += 1
            elif pred == "yes": fp += 1

        result_item = {
            "question_id": qid, "image": img_file, "question": question,
            "gt": label, "pred": pred, "response": response,
        }
        details.append(result_item)
        ans_file.write(json.dumps(result_item, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / total if total else 0.0
    yes_ratio = yes_cnt / total if total else 0.0

    summary = {
        "pope_file": args.pope_file,
        "model_path": args.model_path,
        "generation": {
            "steps": args.steps, "gen_length": args.gen_length,
            "block_length": args.block_length, "temperature": args.temperature,
            "cfg_scale": args.cfg_scale, "remasking": args.remasking,
        },
        "intervention": {
            "num_heads": args.num_heads, "alpha": args.alpha, "beta": args.beta,
            "type": args.type, "head_selection": "probe_auc",
            "direction": "com",
            "mask_scope": "gen_region_mask",
        },
        "total": total, "precision": precision, "recall": recall,
        "f1": f1, "accuracy": acc, "yes_ratio": yes_ratio,
        "counts": {"true_pos": tp, "true_neg": tn, "false_pos": fp, "false_neg": fn, "unknown": unk},
    }

    print(f"\n=== Results === Acc={acc:.4f} F1={f1:.4f} Yes={yes_ratio:.4f}")
    print(json.dumps(summary, indent=2))

    for h in hook_handles:
        h.remove()

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output}")
    print(f"JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
