# save as: tools/test_instructvla_ocr.py

import os
import json
import argparse
import textwrap
import traceback
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist import load_vla


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


def choose_infer_dtype() -> torch.dtype:
    """Pick BF16 when supported, otherwise FP16 (e.g. V100)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def center_crop_image_keep_pil(
    image: Image.Image,
    crop_scale: float = 0.9,
    out_size: int = 224,
) -> Image.Image:
    """Approximate deploy-time preprocessing: center crop + resize to 224."""
    w, h = image.size
    new_w = int(w * crop_scale ** 0.5)
    new_h = int(h * crop_scale ** 0.5)
    left = max((w - new_w) // 2, 0)
    top = max((h - new_h) // 2, 0)
    image = image.crop((left, top, left + new_w, top + new_h))
    image = image.resize((out_size, out_size), Image.BICUBIC)
    return image.convert("RGB")


def render_text_on_image(
    image: Image.Image,
    text: str,
    font_size: int = 36,
    margin: int = 12,
    max_chars_per_line: int = 22,
):
    """Render instruction text on a white banner above the image."""
    image = image.convert("RGB")
    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
    wrapped_text = "\n".join(wrapped_lines) if wrapped_lines else text

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        font_name = "DejaVuSans.ttf"
    except Exception:
        font = ImageFont.load_default()
        font_name = "PIL_default"

    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=6)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    new_w = max(image.width, text_w + margin * 2)
    banner_h = text_h + margin * 2

    canvas = Image.new("RGB", (new_w, image.height + banner_h), "white")
    canvas.paste(image, ((new_w - image.width) // 2, banner_h))

    draw = ImageDraw.Draw(canvas)
    x = (new_w - text_w) // 2
    y = margin
    draw.multiline_text(
        (x, y),
        wrapped_text,
        fill="black",
        font=font,
        spacing=6,
        align="center",
    )

    return canvas, font_name


def horizontal_concat(images):
    """Concatenate images horizontally, top-aligned."""
    images = [im.convert("RGB") for im in images]
    total_w = sum(im.width for im in images)
    max_h = max(im.height for im in images)
    canvas = Image.new("RGB", (total_w, max_h), "white")
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def build_prompt(user_text, image_np):
    """Build prompt in the same role-based format used elsewhere in project."""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": user_text,
            "image": [{"np_array": image_np}],
        },
    ]


def infer_action_shape_args(
    model_path: str,
    action_dim=None,
    future_action_window_size=None,
    past_action_window_size=None,
):
    """Infer action head shape args from run config unless explicitly provided."""
    inferred = {
        "action_dim": action_dim,
        "future_action_window_size": future_action_window_size,
        "past_action_window_size": past_action_window_size,
    }

    ckpt_abs = os.path.abspath(model_path)
    run_dir = os.path.dirname(os.path.dirname(ckpt_abs))
    config_path = os.path.join(run_dir, "config.json")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if inferred["action_dim"] is None:
            inferred["action_dim"] = cfg.get("action_dim")
        if inferred["future_action_window_size"] is None:
            inferred["future_action_window_size"] = cfg.get("future_action_window_size")
        if inferred["past_action_window_size"] is None:
            inferred["past_action_window_size"] = cfg.get("past_action_window_size")

    if inferred["action_dim"] is None:
        inferred["action_dim"] = 7
    if inferred["future_action_window_size"] is None:
        inferred["future_action_window_size"] = 7
    if inferred["past_action_window_size"] is None:
        inferred["past_action_window_size"] = 0

    return inferred


def debug_inputs(inputs):
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"[DEBUG] {k}.shape = {tuple(v.shape)}, dtype = {v.dtype}")
        else:
            print(f"[DEBUG] {k} = {type(v)}")


@torch.no_grad()
def ask_model(model, prompt_messages, max_new_tokens=128, debug=False):
    inputs = model.processor.prepare_input(dict(prompt=prompt_messages))

    if debug:
        debug_inputs(inputs)

    dtype = choose_infer_dtype()
    print(f"[Infer dtype] {dtype}")

    input_ids = inputs["input_ids"].cuda()
    pixel_values = inputs["pixel_values"].cuda()

    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = input_ids.ne(-10)
    attention_mask = attention_mask.cuda()

    if debug:
        print(f"[DEBUG] input_ids.shape: {tuple(input_ids.shape)}")
        print(f"[DEBUG] attention_mask.shape: {tuple(attention_mask.shape)}")
        print(f"[DEBUG] pixel_values.shape: {tuple(pixel_values.shape)}")

    with torch.autocast("cuda", dtype=dtype, enabled=True):
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
    parser.add_argument("--primary_image", type=str, required=True)
    parser.add_argument("--wrist_image", type=str, default=None)
    parser.add_argument("--instruction", type=str, default="open the drawer")
    parser.add_argument("--instruction_file", type=str, default=None)

    parser.add_argument("--use_center_crop", action="store_true")
    parser.add_argument("--render_text", action="store_true")
    parser.add_argument("--concat_wrist", action="store_true")

    parser.add_argument("--save_debug_image", type=str, default="debug_ocr_input.png")
    parser.add_argument("--font_size", type=int, default=36)
    parser.add_argument("--max_chars_per_line", type=int, default=22)

    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--future_action_window_size", type=int, default=None)
    parser.add_argument("--past_action_window_size", type=int, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--single_question", type=str, default=None)

    args = parser.parse_args()

    instruction = args.instruction
    if args.instruction_file is not None:
        with open(args.instruction_file, "r", encoding="utf-8") as f:
            instruction = f.read().strip()

    shape_args = infer_action_shape_args(
        args.model_path,
        action_dim=args.action_dim,
        future_action_window_size=args.future_action_window_size,
        past_action_window_size=args.past_action_window_size,
    )
    print(
        "[Action head args] "
        f"action_dim={shape_args['action_dim']}, "
        f"future_action_window_size={shape_args['future_action_window_size']}, "
        f"past_action_window_size={shape_args['past_action_window_size']}"
    )

    infer_dtype = choose_infer_dtype()
    print(f"[Model dtype target] {infer_dtype}")

    model = load_vla(
        args.model_path,
        load_for_training=False,
        future_action_window_size=shape_args["future_action_window_size"],
        past_action_window_size=shape_args["past_action_window_size"],
        action_dim=shape_args["action_dim"],
    )
    model = model.to(device="cuda", dtype=infer_dtype).eval()

    primary = Image.open(args.primary_image).convert("RGB")
    wrist = Image.open(args.wrist_image).convert("RGB") if args.wrist_image else None

    if args.use_center_crop:
        primary = center_crop_image_keep_pil(primary)
        if wrist is not None:
            wrist = center_crop_image_keep_pil(wrist)

    font_name = None
    if args.render_text:
        primary, font_name = render_text_on_image(
            primary,
            instruction,
            font_size=args.font_size,
            max_chars_per_line=args.max_chars_per_line,
        )

    if args.concat_wrist and wrist is not None:
        final_img = horizontal_concat([primary, wrist])
    else:
        final_img = primary

    final_img.save(args.save_debug_image)
    print(f"[Saved debug image] {args.save_debug_image}")
    if font_name is not None:
        print(f"[Font used] {font_name}")
    print(f"[Final image size] {final_img.size}")

    image_np = np.asarray(final_img)

    tests = [
        ("[Q1] OCR only", "Read the text in the image exactly. Output only the text."),
        ("[Q2] OCR + paraphrase", "What command is written in the image?"),
        ("[Q3] scene understanding", "Describe the image briefly, including any visible text."),
        (
            "[Q4] action from image text",
            "Use the command written in the image as the primary instruction. What should the robot do?",
        ),
    ]

    if args.single_question is not None:
        tests = [("[Single question]", args.single_question)]

    for title, q in tests:
        print("=" * 80)
        print(title)
        print("USER:", q)
        messages = build_prompt(q, image_np)

        try:
            resp = ask_model(
                model,
                messages,
                max_new_tokens=args.max_new_tokens,
                debug=args.debug,
            )
            print("MODEL:", resp)
        except Exception as e:
            print("[ERROR] generation failed")
            print(type(e).__name__, str(e))
            if args.debug:
                traceback.print_exc()
            break


if __name__ == "__main__":
    main()
