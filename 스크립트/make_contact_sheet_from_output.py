from __future__ import annotations

import argparse
from pathlib import Path
import json
from PIL import Image, ImageDraw


def make_contact_sheet(image_paths, out_path, cols=3, thumb_size=(384,384)):
    valid = []
    for p in image_paths:
        pp = Path(str(p))
        if pp.exists():
            valid.append(pp)

    if not valid:
        raise ValueError("No valid image paths found.")

    rows = (len(valid) + cols - 1) // cols
    header_h = 54
    canvas = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "DCEE-CausalVerse Contact Sheet", fill=(0,0,0))

    for idx, p in enumerate(valid):
        img = Image.open(p).convert("RGB")
        img.thumbnail((thumb_size[0]-20, thumb_size[1]-44))
        x0 = (idx % cols) * thumb_size[0]
        y0 = header_h + (idx // cols) * thumb_size[1]
        draw.rectangle([x0, y0, x0 + thumb_size[0]-1, y0 + thumb_size[1]-1], outline=(180,180,180))
        draw.text((x0+10, y0+10), f"Frame {idx+1}", fill=(0,0,0))
        canvas.paste(img, (x0 + (thumb_size[0]-img.width)//2, y0 + 34 + (thumb_size[1]-44-img.height)//2))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output run directory, e.g. outputs/DCEE_CausalVerse_sad_1")
    args = ap.parse_args()

    out_dir = Path(args.out)
    selected_json = out_dir / "selected_images.json"
    paths = []

    if selected_json.exists():
        data = json.loads(selected_json.read_text(encoding="utf-8"))
        for item in data:
            if isinstance(item, dict) and item.get("image_path"):
                paths.append(item["image_path"])

    if not paths:
        # Fallback: one best-looking first candidate per frame
        paths = sorted(str(p) for p in (out_dir / "frames").glob("frame_*_cand_00.png"))
        ending = sorted(str(p) for p in (out_dir / "ending_candidates").glob("frame_*_cand_00.png"))
        paths.extend(ending[:1])

    make_contact_sheet(paths, out_dir / "contact_sheet.png")


if __name__ == "__main__":
    main()
