from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "publication_results" / "final_paper_fusion" / "figures"
NAMES = [
    ("Glioma", "glioma_faithfulness.png"),
    ("Meningioma", "meningioma_faithfulness.png"),
    ("No tumour", "notumor_faithfulness.png"),
    ("Pituitary", "pituitary_faithfulness.png"),
]


def fit(im: Image.Image, width: int, height: int) -> Image.Image:
    copy = im.convert("RGB")
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return canvas


def main() -> None:
    panel_w, panel_h, label_h, gap = 1500, 930, 70, 30
    sheet = Image.new("RGB", (2 * panel_w + 3 * gap, 2 * (panel_h + label_h) + 3 * gap), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, filename) in enumerate(NAMES):
        row, col = divmod(idx, 2)
        x = gap + col * (panel_w + gap)
        y = gap + row * (panel_h + label_h + gap)
        with Image.open(FIG / filename) as im:
            panel = fit(im, panel_w, panel_h)
        sheet.paste(panel, (x, y + label_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (panel_w - (bbox[2] - bbox[0])) / 2, y + 8), label, fill="#1F2933", font=font)

    png = FIG / "figure_11_gradcam_audit.png"
    pdf = FIG / "figure_11_gradcam_audit.pdf"
    sheet.save(png, dpi=(300, 300), optimize=True)
    sheet.save(pdf, resolution=300.0)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
