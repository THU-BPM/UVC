#!/usr/bin/env python3
"""
    Extract instance-scale degraded activation vectors for Lumina-DiMOO.

    - Only YES samples
    - Question: "What is [object] in the image?" (object name IN prompt)
    - Pre-generated instance-scale degraded image from degraded outputs
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
import numpy as np
from PIL import Image

# path setup
_script_dir = os.path.dirname(os.path.abspath(__file__))
_work_dir   = os.path.dirname(os.path.dirname(_script_dir))

_lumina_root_env = os.environ.get("LUMINA_ROOT")
_lumina_candidates = []
if _lumina_root_env:
    _lumina_candidates.append(_lumina_root_env)
_lumina_candidates.append(os.path.join(_work_dir, "Lumina-DiMOO"))
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
from transformers import AutoTokenizer, set_seed
from diffusers import VQModel

# constants
NUM_LAYERS = 32
NUM_HEADS  = 32
HEADS = [f"model.transformer.blocks.{i}.attn_out" for i in range(NUM_LAYERS)]
NEW_LINE = SPECIAL_TOKENS["newline_token"]
MASK_ID  = SPECIAL_TOKENS["mask_token"]  # 126336


def eval_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Lumina-DiMOO from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    vae_ckpt = args.vae_ckpt or args.model_path
    vqvae = VQModel.from_pretrained(vae_ckpt, subfolder="vqvae").to(device)
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    object_pattern = re.compile(r"Is there (?:a |an )?(.+?) in the image", re.IGNORECASE)
    yes_samples = [q for q in questions if q.get("label", "").lower() == "yes"]

    gen_length = args.gen_length

    all_head_wise_activations = []

    named_modules_dict = dict(model.named_modules())
    head_layers = []
    for name in HEADS:
        m = named_modules_dict.get(name)
        if m:
            head_layers.append(m)
        else:
            print(f"Module not found: {name}")

    for line in tqdm(yes_samples[:args.length]):
        idx        = line["question_id"]
        image_file = line["image"]
        orig_qs    = line["text"]

        match = object_pattern.search(orig_qs)
        if match:
            object_name = match.group(1).strip()
        else:
            print(f"Warning: Cannot extract object from: {orig_qs}")
            continue

        # general question (no object name in prompt)
        qs = "What is the object in the image?"

        # instance-scale degraded image
        blurred_path = os.path.join(args.blurred_folder, str(idx), args.degraded_filename)
        if not os.path.exists(blurred_path):
            print(f"Warning: Blurred image not found for qid={idx}, skipping")
            continue

        image = Image.open(blurred_path).convert("RGB")

        crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)
        image = var_center_crop(image, crop_size_list=crop_size_list)
        image_w, image_h = image.size
        _, _, tg_h, tg_w = calculate_vq_params(image_h, image_w, vae_scale)
        input_img_token = encode_img_with_breaks(image, vqvae=vqvae)
        img_token = add_break_line(input_img_token, tg_h, tg_w, new_number=NEW_LINE)

        input_prompt = generate_multimodal_understanding_prompt(qs)
        input_ids    = tokenizer(input_prompt)["input_ids"]
        prompt_tokens = input_ids[:-1] + img_token + input_ids[-1:]

        # append GT answer (object name) to prompt
        obj_token_ids = tokenizer.encode(object_name, add_special_tokens=False)
        prompt_tokens = prompt_tokens + obj_token_ids
        prompt_len    = len(prompt_tokens)

        # Gen region = ALL MASK
        gen_ids = [MASK_ID] * gen_length
        full_input = prompt_tokens + gen_ids

        outputs_dict = {}

        def hook_fn(module, inputs):
            if module not in outputs_dict:
                if inputs and isinstance(inputs[0], torch.Tensor):
                    outputs_dict[module] = inputs[0].detach().cpu()
            return inputs

        hook_handles = [l.register_forward_pre_hook(hook_fn) for l in head_layers]

        input_ids_tensor = torch.tensor([full_input], device=device)
        with torch.no_grad():
            model(input_ids_tensor, infer=True)
        for h in hook_handles:
            h.remove()

        attention_output = torch.stack(tuple(outputs_dict.values()), dim=0) \
                               .float().squeeze().numpy()

        gen_region = attention_output[:, prompt_len:, :]
        gen_mean = gen_region.mean(axis=1)
        all_head_wise_activations.append(gen_mean.copy())

    np.save(args.output, all_head_wise_activations)
    print(f"Saved {len(all_head_wise_activations)} instance-scale degraded vectors -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",     type=str, required=True)
    parser.add_argument("--vae_ckpt",       type=str, default=None)
    parser.add_argument("--image_folder",   type=str, required=True)
    parser.add_argument("--blurred_folder", type=str, required=True)
    parser.add_argument("--degraded_filename", type=str, default="black.jpg")
    parser.add_argument("--question_file",  type=str, required=True)
    parser.add_argument("--output",         type=str, required=True)
    parser.add_argument("--length",         type=int, default=1500)
    parser.add_argument("--gen_length",     type=int, default=2)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
