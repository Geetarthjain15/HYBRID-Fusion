from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parents[1] / "figures"
PNG = OUT / "figure_21_verified_ablation_bar_chart.png"
PDF = OUT / "figure_21_verified_ablation_bar_chart.pdf"
if PNG.exists() or PDF.exists():
    raise FileExistsError("Refusing to overwrite an existing Figure 21 output")

W, H = 2400, 1500
img = Image.new("RGB", (W, H), "#F7FAFF")
draw = ImageDraw.Draw(img)
regular, bold = r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"
title_f = ImageFont.truetype(bold, 60)
sub_f = ImageFont.truetype(regular, 28)
label_f = ImageFont.truetype(bold, 28)
small_f = ImageFont.truetype(regular, 24)
value_f = ImageFont.truetype(bold, 28)
note_f = ImageFont.truetype(regular, 23)

def center(text, y, font, fill):
    b = draw.textbbox((0, 0), text, font=font)
    draw.text(((W - (b[2] - b[0])) / 2, y), text, font=font, fill=fill)

center("Verified Duplicate-Clean Ablation Results", 38, title_f, "#142850")
center("Observed accuracy change relative to the full FusionV2 model (98.27 ± 0.07%)",
       118, sub_f, "#52677F")

names = [
    "ViT branch removed (CNN only)",
    "Auxiliary losses removed",
    "CNN branch removed (ViT only)",
    "Test-time augmentation removed",
    "Multi-scale taps removed",
    "Adaptive gate removed",
    "EMA removed",
]
deltas = [-2.59, -0.55, -0.24, -0.08, -0.04, 0.00, 0.04]
accuracies = ["95.68 ± 0.34%", "97.72 ± 0.48%", "98.03 ± 0.14%",
              "98.19 ± 0.25%", "98.23 ± 0.20%", "98.27 ± 0.18%", "98.31 ± 0.30%"]

plot_left, plot_right = 820, 2200
plot_top, row_h = 245, 145
xmin, xmax = -3.0, 0.5

def xpos(value):
    return int(plot_left + (value - xmin) / (xmax - xmin) * (plot_right - plot_left))

zero_x = xpos(0)
for tick in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]:
    x = xpos(tick)
    draw.line((x, plot_top - 30, x, plot_top + row_h * len(names) - 20),
              fill="#D8E1EC" if tick != 0 else "#344A63", width=2 if tick != 0 else 5)
    t = f"{tick:+.1f}" if tick != 0 else "0.0"
    b = draw.textbbox((0, 0), t, font=small_f)
    draw.text((x - (b[2] - b[0]) / 2, plot_top + row_h * len(names) - 2), t,
              font=small_f, fill="#41566E")

for i, (name, delta, acc) in enumerate(zip(names, deltas, accuracies)):
    cy = plot_top + i * row_h + 50
    draw.rounded_rectangle((35, cy - 41, 775, cy + 42), radius=15,
                           fill="#EBF1F8" if i % 2 == 0 else "#F3F6FA")
    draw.text((60, cy - 29), name, font=label_f, fill="#24364C")
    draw.text((60, cy + 5), f"Accuracy: {acc}", font=small_f, fill="#61748A")

    end_x = xpos(delta)
    y0, y1 = cy - 29, cy + 30
    if delta < 0:
        start, end = end_x, zero_x
        top_c, bottom_c = ((244, 164, 88), (207, 63, 63)) if i == 0 else ((109, 166, 225), (49, 94, 169))
    elif delta > 0:
        start, end = zero_x, end_x
        top_c, bottom_c = (105, 207, 164), (25, 139, 100)
    else:
        start, end = zero_x - 3, zero_x + 3
        top_c = bottom_c = (113, 126, 143)

    width = max(1, end - start)
    for x in range(start, end + 1):
        r = (x - start) / width
        rgb = tuple(round(top_c[j] * (1 - r) + bottom_c[j] * r) for j in range(3))
        draw.line((x, y0, x, y1), fill=rgb, width=1)
    draw.rounded_rectangle((start, y0, end, y1), radius=13, outline="#26364A", width=3)

    text = f"{delta:+.2f} pp" if delta != 0 else "0.00 pp"
    b = draw.textbbox((0, 0), text, font=value_f)
    tx = end_x - (b[2] - b[0]) - 18 if delta < -0.25 else end_x + 18
    draw.text((tx, cy - 18), text, font=value_f,
              fill="#172A45")
    if i == 0:
        star = "Holm-significant at 3/3 seeds"
        b = draw.textbbox((0, 0), star, font=small_f)
        draw.rounded_rectangle((plot_right - (b[2] - b[0]) - 40, cy - 38, plot_right, cy + 38),
                               radius=14, fill="#E7F6F0", outline="#16865F", width=2)
        draw.text((plot_right - (b[2] - b[0]) - 20, cy - 14), star, font=small_f, fill="#116847")

axis = "Accuracy change from full model (percentage points)"
b = draw.textbbox((0, 0), axis, font=label_f)
draw.text(((plot_left + plot_right - (b[2] - b[0])) / 2,
           plot_top + row_h * len(names) + 48), axis, font=label_f, fill="#24364C")

draw.rounded_rectangle((190, 1370, 2210, 1445), radius=18, fill="#EAF1FA", outline="#B6C7DA", width=2)
note = "Bars show observed mean differences across seeds 42, 43 and 44. Only ViT removal survived Holm correction at every seed."
b = draw.textbbox((0, 0), note, font=note_f)
draw.text(((W - (b[2] - b[0])) / 2, 1395), note, font=note_f, fill="#50637A")

img.save(PNG, dpi=(300, 300))
img.save(PDF, "PDF", resolution=300.0)
print(PNG)
print(PDF)
