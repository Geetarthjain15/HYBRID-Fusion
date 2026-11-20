from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "figures"
PNG = OUT / "figure_20_performance_comparison_gradient.png"
PDF = OUT / "figure_20_performance_comparison_gradient.pdf"
if PNG.exists() or PDF.exists():
    raise FileExistsError("Refusing to overwrite an existing Figure 20 output")

W, H = 2400, 1400
img = Image.new("RGB", (W, H), "#F7FAFF")
draw = ImageDraw.Draw(img)
regular, bold = r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"
font_title = ImageFont.truetype(bold, 62)
font_subtitle = ImageFont.truetype(regular, 29)
font_axis = ImageFont.truetype(bold, 30)
font_label = ImageFont.truetype(bold, 31)
font_value = ImageFont.truetype(bold, 44)
font_sd = ImageFont.truetype(regular, 27)
font_note = ImageFont.truetype(regular, 24)
font_callout = ImageFont.truetype(bold, 27)

def centered(text, y, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((W - (box[2] - box[0])) / 2, y), text, font=font, fill=fill)

centered("FusionV2 Accuracy Across Evaluation Regimes", 45, font_title, "#142850")
centered("Strong duplicate-clean internal performance with a clear cross-source generalisation gap",
         125, font_subtitle, "#52677F")

left, right, top, bottom = 250, 2250, 240, 1120
draw.line((left, top, left, bottom), fill="#8297AF", width=4)
draw.line((left, bottom, right, bottom), fill="#8297AF", width=4)
for value in range(0, 101, 10):
    y = bottom - int((value / 100) * (bottom - top))
    draw.line((left, y, right, y), fill="#D7E1ED", width=2)
    text = str(value)
    box = draw.textbbox((0, 0), text, font=font_note)
    draw.text((left - 25 - (box[2] - box[0]), y - 14), text, font=font_note, fill="#30445D")

axis_layer = Image.new("RGBA", (300, 1000), (0, 0, 0, 0))
ImageDraw.Draw(axis_layer).text((20, 450), "Accuracy (%)", font=font_axis, fill="#24364C")
axis_layer = axis_layer.rotate(90, expand=True)
img.paste(axis_layer, (30, 290), axis_layer)

labels = ["Original\nheld-out split", "Duplicate-clean\nheld-out split", "Duplicate-disjoint\ncross-source"]
means, stds = [99.14, 98.27, 69.53], [0.10, 0.07, 1.98]
sample_sizes = ["n = 1,052", "n = 848", "n = 373"]
centres, bar_width = [600, 1250, 1900], 380
colors = [("#2358D8", "#77B9FF"), ("#087F86", "#59D9C6"), ("#D94A3A", "#FFB36B")]

for centre, mean, sd, label, sample, (bottom_c, top_c) in zip(centres, means, stds, labels, sample_sizes, colors):
    bar_top = bottom - int((mean / 100) * (bottom - top))
    x0, x1 = centre - bar_width // 2, centre + bar_width // 2
    c0 = tuple(int(bottom_c[i:i+2], 16) for i in (1, 3, 5))
    c1 = tuple(int(top_c[i:i+2], 16) for i in (1, 3, 5))
    for y in range(bar_top, bottom):
        ratio = (y - bar_top) / max(1, bottom - bar_top)
        rgb = tuple(round(c1[j] * (1 - ratio) + c0[j] * ratio) for j in range(3))
        draw.line((x0, y, x1, y), fill=rgb, width=1)
    draw.rounded_rectangle((x0, bar_top, x1, bottom), radius=18, outline="#26364A", width=4)
    err = max(3, int((sd / 100) * (bottom - top)))
    draw.line((centre, bar_top - err, centre, bar_top + err), fill="#172333", width=5)
    draw.line((centre - 28, bar_top - err, centre + 28, bar_top - err), fill="#172333", width=5)
    draw.line((centre - 28, bar_top + err, centre + 28, bar_top + err), fill="#172333", width=5)
    value_text = f"{mean:.2f}%"
    box = draw.textbbox((0, 0), value_text, font=font_value)
    draw.text((centre - (box[2] - box[0]) / 2, bar_top + 95), value_text, font=font_value, fill="white")
    sd_text = f"± {sd:.2f} SD"
    box = draw.textbbox((0, 0), sd_text, font=font_sd)
    draw.text((centre - (box[2] - box[0]) / 2, bar_top + 155), sd_text, font=font_sd, fill="white")
    for offset, line in enumerate(label.split("\n")):
        box = draw.textbbox((0, 0), line, font=font_label)
        draw.text((centre - (box[2] - box[0]) / 2, bottom + 35 + offset * 38), line, font=font_label, fill="#24364C")
    box = draw.textbbox((0, 0), sample, font=font_note)
    draw.text((centre - (box[2] - box[0]) / 2, bottom - 58), sample, font=font_note, fill="white")

callout = "28.74 percentage-point difference"
box = draw.textbbox((0, 0), callout, font=font_callout)
cx, cy = 1450, 420
draw.rounded_rectangle((cx - 25, cy - 16, cx + (box[2] - box[0]) + 25, cy + 55), radius=16,
                       fill="#FFF0ED", outline="#D65B4A", width=3)
draw.text((cx, cy), callout, font=font_callout, fill="#A63737")
draw.line((1810, 490, 1890, 555), fill="#A63737", width=4)
draw.polygon([(1890, 555), (1868, 546), (1878, 529)], fill="#A63737")

draw.rounded_rectangle((250, 1280, 2250, 1350), radius=20, fill="#EAF1FA", outline="#B6C7DA", width=2)
note = "Bars show means across seeds 42, 43 and 44; error bars are ±1 sample standard deviation, not confidence intervals."
box = draw.textbbox((0, 0), note, font=font_note)
draw.text(((W - (box[2] - box[0])) / 2, 1301), note, font=font_note, fill="#50637A")

img.save(PNG, dpi=(300, 300))
img.save(PDF, "PDF", resolution=300.0)
print(PNG)
print(PDF)
