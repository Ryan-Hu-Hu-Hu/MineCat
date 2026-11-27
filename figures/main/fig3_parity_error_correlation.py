"""
Figure 3 - composite of (A) parity plots and (B) error-correlation heatmaps.

Runs fig3a_parity.py and fig3b_error_correlation.py, then stacks their PNGs.
Output: figures/main/fig3_parity_error_correlation.{png,pdf}
Run:    python figures/main/fig3_parity_error_correlation.py
"""
import os
import runpy

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

FIGDIR = os.path.dirname(os.path.abspath(__file__))
for script in ("fig3a_parity.py", "fig3b_error_correlation.py"):
    runpy.run_path(os.path.join(FIGDIR, script), run_name="__main__")
parity = Image.open(os.path.join(FIGDIR, "fig3a_parity.png")).convert("RGBA")
corr   = Image.open(os.path.join(FIGDIR, "fig3b_error_correlation.png")).convert("RGBA")

# common width
W = max(parity.width, corr.width)
def fit(im):
    if im.width == W:
        return im
    h = round(im.height * W / im.width)
    return im.resize((W, h), Image.LANCZOS)
parity, corr = fit(parity), fit(corr)

PAD_TOP = 150         # room for panel label above each sub-panel
GAP     = 40          # gap between the two panels
LABEL_X = 30
H = PAD_TOP + parity.height + GAP + PAD_TOP + corr.height
canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

y = 0
# panel a
y_a_label = y
canvas.paste(parity, (0, y + PAD_TOP), parity)
y = y + PAD_TOP + parity.height + GAP
# panel b
y_b_label = y
canvas.paste(corr, (0, y + PAD_TOP), corr)

draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype(font_manager.findfont(
        font_manager.FontProperties(family="Arial", weight="bold")), 185)
except Exception:
    font = ImageFont.load_default()
draw.text((LABEL_X, y_a_label + 10), "A", fill=(0, 0, 0, 255), font=font)
draw.text((LABEL_X, y_b_label + 10), "B", fill=(0, 0, 0, 255), font=font)

out_png = os.path.join(FIGDIR, "fig3_parity_error_correlation.png")
canvas.convert("RGB").save(out_png, dpi=(300, 300))
# also a PDF for completeness
canvas.convert("RGB").save(os.path.join(FIGDIR, "fig3_parity_error_correlation.pdf"), "PDF", resolution=300)
print("saved", out_png, canvas.size)
