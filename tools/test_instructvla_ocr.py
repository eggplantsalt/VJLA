# save as: tools/test_instructvla_ocr.py

import os
import argparse
import textwrap
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist import load_vla


def center_crop_image_keep_pil(image: Image.Image, crop_scale: float = 0.9, out_size: int = 224) -> Image.Image:
    """尽量复刻 deploy/instructvla_utils.py 的评测前处理：center crop + resize to 224."""
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
    """把指令渲染到图像顶部白底条上。"""
    image = image.convert("RGB")
    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
    wrapped_text = "\n".join(wrapped_lines)

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
    draw.multiline_text((x, y), wrapped_text, fill="black", font=font, spacing=6, align="center")

    return canvas, font_name


def horizontal_concat(images):
    """横向拼接多张图，顶部对齐。"""
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
    return [
        {"content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": user_text,
            "image": [{"np_array": image_np}],
        },
    ]


@torch.no_grad()
def ask_model(model, prompt_messages, max_new_tokens=128):
    inputs = model.processor.prepare_input(dict(prompt=prompt_messages))
    dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=True):
        output = model.vlm.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=max_new_tokens,
            output_hidden_states=False,
        )

    response = model.processor.tokenizer.decode(output[0], skip_special_tokens=False)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--primary_image", type=str, required=True)
    parser.add_argument("--wrist_image", type=str, default=None)
    parser.add_argument("--instruction", type=str, default="open the drawer")
    parser.add_argument("--use_center_crop", action="store_true")
    parser.add_argument("--render_text", action="store_true")
    parser.add_argument("--concat_wrist", action="store_true")
    parser.add_argument("--save_debug_image", type=str, default="debug_ocr_input.png")
    parser.add_argument("--font_size", type=int, default=36)
    parser.add_argument("--max_chars_per_line", type=int, default=22)
    parser.add_argument("--instruction_file", type=str, default=None)
    args = parser.parse_args()
    instruction = args.instruction
    if args.instruction_file is not None:
        with open(args.instruction_file, "r", encoding="utf-8") as f:
            instruction = f.read().strip()

    model = load_vla(
        args.model_path,
        load_for_training=False,
        future_action_window_size=7,
        past_action_window_size=0,
        action_dim=7,
    )
    model = model.to(device="cuda", dtype=torch.float32).eval()

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

    for title, q in tests:
        print("=" * 80)
        print(title)
        print("USER:", q)
        messages = build_prompt(q, image_np)
        resp = ask_model(model, messages)
        print("MODEL:", resp)


if __name__ == "__main__":
    main()

