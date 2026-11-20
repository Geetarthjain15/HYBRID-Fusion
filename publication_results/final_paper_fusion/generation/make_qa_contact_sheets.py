from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
QA = ROOT / "publication_results" / "final_paper_fusion" / "qa"


def sheet(folder: Path, output: Path, cols: int = 4, thumb_w: int = 340) -> None:
    paths = sorted(folder.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
    if not paths:
        raise RuntimeError(f"No rendered pages in {folder}")
    with Image.open(paths[0]) as first:
        ratio = first.height / first.width
    thumb_h = round(thumb_w * ratio)
    label_h, gap = 34, 18
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * gap,
                               rows * (thumb_h + label_h) + (rows + 1) * gap), "#D9DEE5")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for idx, path in enumerate(paths):
        row, col = divmod(idx, cols)
        x = gap + col * (thumb_w + gap)
        y = gap + row * (thumb_h + label_h + gap)
        draw.text((x, y), f"Page {idx + 1}", fill="#1F2933", font=font)
        with Image.open(path) as im:
            page = im.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(page, (x, y + label_h))
    canvas.save(output, optimize=True)


def main() -> None:
    sheet(QA / "manuscript_pages", QA / "manuscript_contact_sheet.png")
    sheet(QA / "supplement_pages", QA / "supplement_contact_sheet.png", cols=3)


if __name__ == "__main__":
    main()
