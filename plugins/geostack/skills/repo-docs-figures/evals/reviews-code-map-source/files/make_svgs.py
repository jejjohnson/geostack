"""Draw krigax's import graph.

Run: python make_svgs.py --out docs/assets/imports.svg
Preview the result with cairosvg before committing.
"""

import argparse

TITLE = "Import graph"

# module -> (public names, note)
CARDS = {
    "_kernels": (["RBF", "Matern32"], "kernels"),
    "_solvers": (["CholeskySolver", "CGSolver"], "CGSolver is matrix-free: K is never formed"),
    "_model": (["GaussianProcess", "fit"], "the fitted model"),
    "_predict": (["predict", "log_marginal_likelihood"], "algorithms"),
    "_sparse": (["SparseGP"], "inducing points"),
}

# (from, to, style)
EDGES = [
    ("_model", "_solvers", "solid"),
    ("_predict", "_model", "solid"),
    ("_predict", "_solvers", "solid"),
    ("_sparse", "_model", "solid"),
    ("_solvers", "lineax", "solid"),
    ("_model", "lineax", "solid"),
    ("_predict", "lineax", "solid"),
    ("_predict", "einx", "dashed"),  # through an extra
    ("_kernels", "equinox", "solid"),
    ("_solvers", "equinox", "solid"),
    ("_model", "equinox", "solid"),
    ("_model", "matplotlib", "solid"),
]

PALETTE = {"bg": "#ffffff", "ink": "#1f2328", "card": "#f6f8fa", "edge": "#57606a"}


def draw(out):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520">']
    parts.append(f'<rect width="900" height="520" fill="{PALETTE["bg"]}"/>')
    parts.append(f'<text x="24" y="36" fill="{PALETTE["ink"]}">{TITLE}</text>')
    for i, (name, (chips, note)) in enumerate(CARDS.items()):
        y = 70 + 84 * i
        parts.append(f'<rect x="24" y="{y}" width="420" height="68" fill="{PALETTE["card"]}"/>')
        parts.append(f'<text x="36" y="{y + 24}">{name}: {", ".join(chips)}</text>')
        parts.append(f'<text x="36" y="{y + 48}">{note}</text>')
    parts.append("</svg>")
    with open(out, "w") as f:
        f.write("\n".join(parts))
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/assets/imports.svg")
    draw(parser.parse_args().out)
