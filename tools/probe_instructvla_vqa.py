import argparse
import importlib
import json
import os
import traceback
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


def choose_infer_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_image(image_path: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image


def build_prompt(user_text: str, image_np: np.ndarray, system_text: str = DEFAULT_SYSTEM_MESSAGE):
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": user_text,
            "image": [{"np_array": image_np}],
        },
    ]


def maybe_read_text(text: str | None, text_file: str | None) -> str:
    if text_file is not None:
        with open(text_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    if text is None:
        raise ValueError("You must provide either --question or --question_file")
    return text


def maybe_read_questions(question: str | None, question_file: str | None, questions_json: str | None):
    if questions_json is not None:
        with open(questions_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("--questions_json must contain a JSON list of strings")
        return data

    return [maybe_read_text(question, question_file)]


def import_loader(loader_module: str, loader_fn: str):
    module = importlib.import_module(loader_module)
    fn = getattr(module, loader_fn)
    return fn


def infer_action_shape_args_from_config(model_path: str):
    """
    尽量从 checkpoint 相邻的 config.json 里推 action head 参数。
    对纯 VQA 不一定必须，但某些 load_vla 需要这些参数。
    """
    inferred = {
        "action_dim": None,
        "future_action_window_size": None,
        "past_action_window_size": None,
    }

    ckpt_abs = os.path.abspath(model_path)
    run_dir = os.path.dirname(os.path.dirname(ckpt_abs))
    config_path = os.path.join(run_dir, "config.json")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        inferred["action_dim"] = cfg.get("action_dim")
        inferred["future_action_window_size"] = cfg.get("future_action_window_size")
        inferred["past_action_window_size"] = cfg.get("past_action_window_size")

    return inferred


def load_model(
    model_path: str,
    loader_module: str,
    loader_fn: str,
    device: str = "cuda",
):
    load_fn = import_loader(loader_module, loader_fn)

    shape_args = infer_action_shape_args_from_config(model_path)

    load_kwargs: dict[str, Any] = {
        "load_for_training": False,
    }

    if shape_args["action_dim"] is not None:
        load_kwargs["action_dim"] = shape_args["action_dim"]
    if shape_args["future_action_window_size"] is not None:
        load_kwargs["future_action_window_size"] = shape_args["future_action_window_size"]
    if shape_args["past_action_window_size"] is not None:
        load_kwargs["past_action_window_size"] = shape_args["past_action_window_size"]

    print("[Loader module]", loader_module)
    print("[Loader function]", loader_fn)
    print("[Model path]", model_path)
    print("[Inferred kwargs]", load_kwargs)

    model = load_fn(model_path, **load_kwargs)

    infer_dtype = choose_infer_dtype()
    print("[Model dtype target]", infer_dtype)

    model = model.to(device=device, dtype=infer_dtype).eval()
    return model, infer_dtype


def debug_inputs(inputs: dict[str, Any]):
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"[DEBUG] {k}.shape = {tuple(v.shape)}, dtype = {v.dtype}")
        else:
            print(f"[DEBUG] {k} = {type(v)}")


@torch.no_grad()
def ask_model(
    model,
    image_np: np.ndarray,
    question: str,
    max_new_tokens: int = 128,
    debug: bool = False,
    system_text: str = DEFAULT_SYSTEM_MESSAGE,
):
    prompt_messages = build_prompt(question, image_np, system_text=system_text)
    inputs = model.processor.prepare_input(dict(prompt=prompt_messages))

    if debug:
        debug_inputs(inputs)

    input_ids = inputs["input_ids"].cuda()

    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = input_ids.ne(-10)
    attention_mask = attention_mask.cuda()

    pixel_values = inputs["pixel_values"].cuda()

    infer_dtype = choose_infer_dtype()
    print("[Infer dtype]", infer_dtype)

    if debug:
        print(f"[DEBUG] input_ids.shape: {tuple(input_ids.shape)}")
        print(f"[DEBUG] attention_mask.shape: {tuple(attention_mask.shape)}")
        print(f"[DEBUG] pixel_values.shape: {tuple(pixel_values.shape)}")

    with torch.autocast("cuda", dtype=infer_dtype, enabled=True):
        output = model.vlm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            output_hidden_states=False,
            return_dict_in_generate=False,
        )

    response = model.processor.tokenizer.decode(output[0], skip_special_tokens=False)
    return response


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--loader_module",
        type=str,
        required=True,
        help="e.g. vla.instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist",
    )
    parser.add_argument(
        "--loader_fn",
        type=str,
        default="load_vla",
        help="e.g. load_vla",
    )

    parser.add_argument("--image_path", type=str, required=True)

    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--question_file", type=str, default=None)
    parser.add_argument(
        "--questions_json",
        type=str,
        default=None,
        help="JSON file containing a list of questions",
    )

    parser.add_argument("--system_text", type=str, default=DEFAULT_SYSTEM_MESSAGE)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    questions = maybe_read_questions(
        question=args.question,
        question_file=args.question_file,
        questions_json=args.questions_json,
    )

    model, _ = load_model(
        model_path=args.model_path,
        loader_module=args.loader_module,
        loader_fn=args.loader_fn,
    )

    image = load_image(args.image_path)
    image_np = np.asarray(image)

    print("[Image path]", args.image_path)
    print("[Image size]", image.size)
    print("[Num questions]", len(questions))

    for i, q in enumerate(questions, start=1):
        print("=" * 80)
        print(f"[Question {i}]")
        print("USER:", q)
        try:
            response = ask_model(
                model=model,
                image_np=image_np,
                question=q,
                max_new_tokens=args.max_new_tokens,
                debug=args.debug,
                system_text=args.system_text,
            )
            print("MODEL:", response)
        except Exception as e:
            print("[ERROR] generation failed")
            print(type(e).__name__, str(e))
            if args.debug:
                traceback.print_exc()
            break


if __name__ == "__main__":
    main()
