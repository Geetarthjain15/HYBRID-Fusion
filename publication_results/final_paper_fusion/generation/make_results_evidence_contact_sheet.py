from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "publication_results" / "final_paper_fusion" / "qa" / "results_evidence_guide"
PAGES = sorted(FOLDER.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
if not PAGES:
    raise RuntimeError("No rendered guide pages found")

cols, thumb_w, label_h, gap = 3, 360, 34, 18
with Image.open(PAGES[0]) as first:
    thumb_h = round(thumb_w * first.height / first.width)
rows = (len(PAGES) + cols - 1) // cols
sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * gap,
                          rows * (thumb_h + label_h) + (rows + 1) * gap), "#D9DEE5")
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 22)
except OSError:
    font = ImageFont.load_default()

for index, path in enumerate(PAGES):
    row, col = divmod(index, cols)
    x = gap + col * (thumb_w + gap)
    y = gap + row * (thumb_h + label_h + gap)
    draw.text((x, y), f"Page {index + 1}", fill="#1F2933", font=font)
    with Image.open(path) as page:
        image = page.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    sheet.paste(image, (x, y + label_h))

sheet.save(FOLDER / "contact_sheet.png", optimize=True)
