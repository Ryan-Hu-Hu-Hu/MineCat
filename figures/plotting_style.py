"""Shared matplotlib style for the MineCat figures (Arial, bold labels, 300-dpi PNG + PDF).

Fonts / colours / DPI can be tweaked in figure_style.yaml without touching the scripts.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_STYLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_style.yaml")


def _load_style():
    try:
        import yaml
        with open(_STYLE_PATH) as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


_STYLE = _load_style()

if os.environ.get("FIG_SVG") == "1":           # keep text editable in SVG output
    matplotlib.rcParams["svg.fonttype"] = "none"


def _cfg(section, key, default):
    v = _STYLE.get(section, {}).get(key, None) if isinstance(_STYLE.get(section), dict) else None
    return default if v is None else v


def use_arial():
    """Arial for every text element (falls back to the default sans-serif if absent)."""
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "Arial"
    matplotlib.rcParams["mathtext.it"] = "Arial:italic"
    matplotlib.rcParams["mathtext.bf"] = "Arial:bold"


AXIS_LABEL_FONTSIZE = _cfg("fonts", "axis_label", 24)
TICK_LABEL_FONTSIZE = _cfg("fonts", "tick_label", 20)
TITLE_FONTSIZE = _cfg("fonts", "title", 26)
LEGEND_FONTSIZE = _cfg("fonts", "legend", 20)
VALUE_LABEL_FONTSIZE = _cfg("fonts", "value_label", 20)

COLORS = {
    "MineCat (deep)": "#E63946",
    "MineCat (ensemble)": "#7A1128",
    "MineCat (tree)": "#B5651D",
    "MPEK": "#8E44AD",
    "EF-UniKP": "#2A9D8F",
    "PreTKcat": "#F4A261",
    "DLTKcat": "#264653",
    "CataPro": "#457B9D",
    "KinForm": "#1D3557",
    "KmPred": "#E9C46A",
    "UniKP": "#A8DADC",
    "CatPred": "#9B2226",
}
if isinstance(_STYLE.get("colors"), dict):
    COLORS.update({k: v for k, v in _STYLE["colors"].items() if v})

# the three ensemble members (used wherever tree / deep / composition are compared)
MEMBER_COLORS = {"MineCat (tree)": "#C44E52", "MineCat (deep)": "#4C72B0",
                 "MineCat (composition)": "#2A9D8F"}

METRIC_LABELS = {"R2": "R²", "PCC": "PCC", "SCC": "SCC", "MAE": "MAE", "RMSE": "RMSE"}


def add_side_legend(ax, labels, colors=None, title=None):
    colors = colors or {lab: COLORS.get(lab, "#333333") for lab in labels}
    handles = [Patch(facecolor=colors[lab], label=lab, edgecolor="grey") for lab in labels]
    return ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, title=title,
                     title_fontsize=LEGEND_FONTSIZE, prop={"weight": "bold"},
                     frameon=_cfg("legend", "frameon", True),
                     loc=_cfg("legend", "loc", "upper left"),
                     bbox_to_anchor=tuple(_cfg("legend", "bbox_to_anchor", (1.01, 1.0))),
                     borderaxespad=0)


def savefig(fig, out_dir, name, dpi=None):
    """Save <out_dir>/<name>.png and .pdf (and .svg when FIG_SVG=1), then close."""
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(bbox_inches=_cfg("savefig", "bbox_inches", "tight"),
              transparent=_cfg("savefig", "transparent", False), facecolor="white")
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=dpi or _cfg("savefig", "png_dpi", 300), **kw)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), dpi=dpi or _cfg("savefig", "pdf_dpi", 300), **kw)
    if os.environ.get("FIG_SVG") == "1":
        fig.savefig(os.path.join(out_dir, f"{name}.svg"), **kw)
    plt.close(fig)
    print(f"saved {name}")
