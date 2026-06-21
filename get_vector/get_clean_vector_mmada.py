#!/usr/bin/env python3
"""
Extract clean-reference activation vectors for MMaDA.

  - Only YES samples
  - Question: "What is the object in the image?"
  - Clean image
  - Gen region = ALL MASK (no GT in gen, matches injection)
  - Extract: mean over gen region from attn_out PRE-HOOK (per-head semantic)
"""
import argparse
import torch
import os
import json
import re
from tqdm import tqdm
import sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_mmada_candidates = []
_mmada_root_env = os.environ.get("MMADA_ROOT")
if _mmada_root_env:
    _mmada_candidates.append(_mmada_root_env)
_mmada_candidates.append(os.path.join(_repo_root, "MMaDA"))
for _p in _mmada_candidates:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
from models import MAGVITv2, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from training.utils import image_transform
from transformers import AutoTokenizer, set_seed
from PIL import Image
import numpy as np


def eval_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    model = MMadaModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    vq_model_path = getattr(args, 'vq_model_path', "showlab/magvitv2")
    vq_model = MAGVITv2.from_pretrained(vq_model_path)
    vq_model.to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    resolution = getattr(args, 'resolution', 512)
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=512,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=0.0,
        use_reserved_token=True
    )

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    all_head_wise_activations = []

    gen_length = getattr(args, 'gen_length', 2)
    MASK_TOKEN_ID = 126336

    object_pattern = re.compile(r"Is there (?:a |an )?(.+?) in the image", re.IGNORECASE)

    # only use YES samples
    yes_questions = [q for q in questions if q["label"] == "yes"]

    # Pre-cache named modules
    named_modules = dict(model.named_modules())
    HEADS = [f"model.transformer.blocks.{i}.attn_out" for i in range(32)]
    head_layers = []
    for name in HEADS:
        module = named_modules.get(name)
        if module:
            head_layers.append(module)
        else:
            print(f"Module not found: {name}")

    for line in tqdm(yes_questions[:args.length]):
        image_file = line["image"]
        orig_qs = line["text"]

        # extract object name from POPE question
        match = object_pattern.search(orig_qs)
        if match:
            object_name = match.group(1).strip()
        else:
            object_name = "object"
            print(f"Warning: Could not extract object from: {orig_qs}")

        # general question (no object name in prompt)
        qs = "What is the object in the image?"

        # clear image
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")

        image_tensor = image_transform(image, resolution=resolution).unsqueeze(0).to(device)
        image_tokens = vq_model.get_code(image_tensor) + len(tokenizer)

        # Construct text tokens
        messages = [{"role": "user", "content": qs}]
        text_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # append GT answer (object name) to prompt
        obj_token_ids = tokenizer.encode(object_name, add_special_tokens=False)
        obj_tensor = torch.tensor([obj_token_ids], dtype=torch.long, device=device)
        text_token_ids = torch.cat([text_token_ids, obj_tensor], dim=1)

        # Gen region = ALL MASK (same as injection)
        gen_ids = torch.full((1, gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device)

        # Prompt: <|mmu|><|soi|>[image_tokens]<|eoi|>[text_tokens]
        prompt_ids = torch.cat([
            torch.tensor([[uni_prompting.sptids_dict['<|mmu|>'].item()]]).to(device),
            torch.tensor([[uni_prompting.sptids_dict['<|soi|>'].item()]]).to(device),
            image_tokens,
            torch.tensor([[uni_prompting.sptids_dict['<|eoi|>'].item()]]).to(device),
            text_token_ids
        ], dim=1).long()
        prompt_len = prompt_ids.shape[1]

        # Full input: prompt + gen_region (ALL MASK)
        input_ids = torch.cat([prompt_ids, gen_ids], dim=1).long()

        outputs_dict = {}

        # IMPORTANT: pre-hook on attn_out INPUT (per-head concat, before linear mixing)
        def hook_fn(module, inputs):
            if module not in outputs_dict:
                if inputs and isinstance(inputs[0], torch.Tensor):
                    outputs_dict[module] = inputs[0].detach().cpu()
            return inputs

        hook_handles = [layer.register_forward_pre_hook(hook_fn) for layer in head_layers]

        with torch.no_grad():
            model(input_ids)

        for handle in hook_handles:
            handle.remove()

        attention_output = tuple(outputs_dict.values())
        attention_output = torch.stack(attention_output, dim=0).float().squeeze().numpy()
        # shape: (num_layers, seq_len, hidden_size)
        gen_region = attention_output[:, prompt_len:, :]  # (layers, gen_len, hidden)
        gen_mean = gen_region.mean(axis=1)
        all_head_wise_activations.append(gen_mean.copy())

    np.save(args.output, all_head_wise_activations)
    print(f"Saved {len(all_head_wise_activations)} clean-reference vectors -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--vq-model-path", type=str, default="showlab/magvitv2")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--length", type=int, default=1500)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gen_length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
