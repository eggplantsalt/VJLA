import argparse
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont


def load_font(font_path: str | None, font_size: int):
    if font_path is not None and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, font_size), os.path.basename(font_path)
        except Exception:
            pass

    # 常见无头环境优先尝试 DejaVuSans
    for candidate in [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, font_size), candidate
        except Exception:
            continue

    return ImageFont.load_default(), "PIL_default"


def make_text_image(
    text: str,
    out_path: str,
    width: int = 224,
    height: int = 224,
    font_size: int = 64,
    margin: int = 60,
    line_spacing: int = 12,
    max_chars_per_line: int = 18,
    align: str = "center",
    font_path: str | None = None,
):
    font, font_name = load_font(font_path, font_size)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line) if text else [""]
    wrapped_text = "\n".join(wrapped_lines)

    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=line_spacing, align=align)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if align == "center":
        x = (width - text_w) // 2
    elif align == "left":
        x = margin
    elif align == "right":
        x = max(width - text_w - margin, margin)
    else:
        raise ValueError(f"Unsupported align: {align}")

    y = (height - text_h) // 2

    draw.multiline_text(
        (x, y),
        wrapped_text,
        fill="black",
        font=font,
        spacing=line_spacing,
        align=align,
    )

    image.save(out_path)
    print(f"[Saved] {out_path}")
    print(f"[Font] {font_name}")
    print(f"[Image size] {(width, height)}")
    print(f"[Text] {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to render")
    parser.add_argument("--out", type=str, default="text_image.png", help="Output image path")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--font_size", type=int, default=64)
    parser.add_argument("--margin", type=int, default=60)
    parser.add_argument("--line_spacing", type=int, default=12)
    parser.add_argument("--max_chars_per_line", type=int, default=18)
    parser.add_argument("--align", type=str, default="center", choices=["left", "center", "right"])
    parser.add_argument("--font_path", type=str, default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", help="Path to .ttf font file")

    args = parser.parse_args()

    make_text_image(
        text=args.text,
        out_path=args.out,
        width=args.width,
        height=args.height,
        font_size=args.font_size,
        margin=args.margin,
        line_spacing=args.line_spacing,
        max_chars_per_line=args.max_chars_per_line,
        align=args.align,
        font_path=args.font_path,
    )


if __name__ == "__main__":
    main()



