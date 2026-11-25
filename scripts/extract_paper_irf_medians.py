"""Read the median impulse responses of Brignone & Piffer (2025) Figures 2-3
out of the vector drawing in the committed PDF (issue #11).

    uv run --with pymupdf python scripts/extract_paper_irf_medians.py

Writes data/external/brignone_piffer_2025_fig2_fig3_irf_medians.csv.

The paper publishes its impulse responses as figures only. In the PDF those
figures are vector drawings, not images: each panel's median is one stroked
polyline with a vertex per quarter (h = 0..20), and each panel has one horizontal line
at zero. So the medians can be read from path coordinates instead of from
pixels. What cannot be read from the file is the text of the y-axis tick
labels, which the PDF stores as glyph outlines. Those values are transcribed
by hand in ``TICKS`` below from a rendering of the two pages, and checked
against the drawing in four ways before any number is written:

1. the number of label rows found left of the axis equals the number of
   transcribed labels;
2. the number of digit glyphs in each row equals the number of digits in the
   transcribed label (so "0.5" read as "5", or "10" as "1", is caught);
3. the label positions are a straight line in the transcribed values (so a
   single misread digit is caught);
4. the zero implied by that line is where the panel's zero line is drawn.

What these checks cannot catch is every label of a panel misread in a way
that keeps the digit counts and the even spacing (say "0, 2, 4" read as
"0, 3, 6"). Each value therefore carries the label step it rests on.

Coordinates in the PDF sit on a 0.12 pt grid, so a value is good to about
half a grid step: ``resolution`` in the output is 0.06 pt in the panel's
units. PyMuPDF is AGPL-licensed, so it is not a project dependency; the
tests read only the CSV (and re-run this script when PyMuPDF happens to be
installed).
"""

from __future__ import annotations

import csv
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
PDF = os.path.join(ROOT, "docs", "a-structural-var-model-for-the-uk-economy.pdf")
OUT = os.path.join(ROOT, "data", "external",
                   "brignone_piffer_2025_fig2_fig3_irf_medians.csv")

# Model column order; also the reading order of the eight panels in each
# block (two rows of four).
VARIABLES = ["world_gdp", "world_cpi", "oil_price", "bank_rate",
             "eri", "cpisa", "cpi_energy", "uk_gdp"]
UNITS = {v: "percent" for v in VARIABLES} | {"bank_rate": "basis points"}

# (figure, 0-based page index, shocks of blocks A, B, C from the top)
PAGES = [
    ("2", 13, ["world_demand", "world_supply", "world_energy"]),
    ("3", 14, ["uk_demand", "uk_supply", "uk_monpol"]),
]

# y-axis tick labels as printed, bottom to top, transcribed from a 250 dpi
# rendering of pages 14 and 15 of the PDF (printed page numbers 12 and 13).
TICKS = {
    "world_demand": [
        ["-0.2", "0", "0.2", "0.4"], ["0", "0.5", "1"],
        ["-5", "0", "5", "10"], ["0", "50"],
        ["-1", "0", "1", "2"], ["0", "0.5", "1", "1.5"],
        ["0", "2", "4"], ["-0.5", "0", "0.5"],
    ],
    "world_supply": [
        ["0", "0.2", "0.4"], ["-1", "-0.5", "0"],
        ["-5", "0", "5", "10"], ["-20", "0", "20"],
        ["-2", "0", "2"], ["-1", "-0.5", "0"],
        ["-1", "0", "1", "2"], ["-0.2", "0", "0.2", "0.4"],
    ],
    "world_energy": [
        ["0", "0.2", "0.4", "0.6"], ["-1", "-0.5", "0"],
        ["-10", "-5", "0"], ["-20", "0", "20"],
        ["0", "1", "2"], ["-1", "-0.5", "0"],
        ["-4", "-2", "0"], ["0", "0.5", "1"],
    ],
    "uk_demand": [
        ["-0.1", "0", "0.1", "0.2"], ["-0.5", "0", "0.5"],
        ["-4", "-2", "0", "2"], ["0", "10", "20", "30"],
        ["-1", "0", "1", "2"], ["0", "0.5", "1"],
        ["-1", "0", "1", "2"], ["0", "0.2", "0.4", "0.6"],
    ],
    "uk_supply": [
        ["-0.1", "0", "0.1", "0.2"], ["-0.5", "0", "0.5"],
        ["-5", "0", "5"], ["-20", "0", "20"],
        ["-2", "0", "2"], ["-1", "-0.5", "0"],
        ["-2", "-1", "0", "1"], ["0", "0.2", "0.4", "0.6"],
    ],
    "uk_monpol": [
        ["-0.1", "0", "0.1", "0.2"], ["-0.4", "-0.2", "0"],
        ["-5", "0", "5"], ["-20", "0", "20"],
        ["-2", "-1", "0", "1"], ["-1", "-0.5", "0"],
        ["-3", "-2", "-1", "0"], ["-0.4", "-0.2", "0", "0.2"],
    ],
}

GRID = 0.12          # pt; the PDF's coordinate grid
TOL = 0.2            # pt; allowed misfit of label positions and of the zero
LABEL_GREY = 0.149   # fill of the axis text


class ExtractionError(RuntimeError):
    pass


def _is(colour, value, tol=0.01):
    return colour is not None and all(abs(c - value) < tol for c in colour)


def _polyline(drawing):
    items = drawing["items"]
    return [items[0][1]] + [it[2] for it in items]


def _panels(drawings):
    """The 24 medians of a page, sorted into reading order."""
    meds = [d for d in drawings
            if d["type"] == "s" and _is(d["color"], 0.0)
            and abs(d["width"] - 1.32) < 0.02 and 10 <= len(d["items"]) <= 20
            and all(it[0] == "l" for it in d["items"])]
    if len(meds) != 24:
        raise ExtractionError(f"expected 24 median lines, found {len(meds)}")
    return meds


def _on_horizons(pts, where):
    """y at h = 0..20. Six medians in Figure 3 have 20 vertices, not 21: the
    PDF writer dropped a vertex that lay on the straight line between its
    neighbours, so linear interpolation restores it exactly (to the grid)."""
    step = (pts[-1].x - pts[0].x) / 20.0
    known = {}
    for p in pts:
        h = round((p.x - pts[0].x) / step)
        if abs(p.x - (pts[0].x + h * step)) > GRID + 0.01 or h in known:
            raise ExtractionError(f"{where}: a vertex is off the quarterly grid")
        known[h] = p.y
    if 0 not in known or 20 not in known or len(known) < 19:
        raise ExtractionError(f"{where}: {len(known)} of 21 vertices found")
    hs = sorted(known)
    out = []
    for h in range(21):
        if h in known:
            out.append(known[h])
            continue
        lo = max(x for x in hs if x < h)
        hi = min(x for x in hs if x > h)
        out.append(known[lo] + (known[hi] - known[lo]) * (h - lo) / (hi - lo))
    return out


def _axis_box(drawings, pts):
    """The white rectangle the panel is drawn on."""
    x0, x1 = pts[0].x, pts[-1].x
    ys = [p.y for p in pts]
    boxes = [d["rect"] for d in drawings
             if d["type"] == "f" and _is(d["fill"], 1.0)
             and len(d["items"]) == 1 and d["items"][0][0] == "re"
             and abs(d["rect"].x0 - x0) < 0.5 and abs(d["rect"].x1 - x1) < 0.5
             and d["rect"].y0 <= min(ys) + 0.5 and d["rect"].y1 >= max(ys) - 0.5]
    if len(boxes) != 1:
        raise ExtractionError(f"expected one axis box, found {len(boxes)}")
    return boxes[0]


def _zero_line(drawings, box):
    lines = []
    for d in drawings:
        if not (d["type"] == "s" and _is(d["color"], 0.0)
                and abs(d["width"] - 0.6) < 0.02 and len(d["items"]) == 1):
            continue
        a, b = d["items"][0][1], d["items"][0][2]
        if (abs(a.y - b.y) < 1e-6 and abs(min(a.x, b.x) - box.x0) < 0.5
                and abs(max(a.x, b.x) - box.x1) < 0.5
                and box.y0 - 0.5 <= a.y <= box.y1 + 0.5):
            lines.append(a.y)
    if len(lines) != 1:
        raise ExtractionError(f"expected one zero line, found {len(lines)}")
    return lines[0]


def _label_rows(drawings, box):
    """y-axis tick labels: [(y centre of the digits, digit count)], bottom up.

    Only digits are counted. Minus signs and decimal points are sometimes
    merged into a compound path with the tick marks, so they cannot be
    counted reliably; a missed one changes the digit count or breaks the
    straight-line check instead.
    """
    glyphs = [d["rect"] for d in drawings
              if d["type"] == "f" and _is(d["fill"], LABEL_GREY)
              and box.x0 - 13.0 <= d["rect"].x0 and d["rect"].x1 <= box.x0 - 1.0
              and box.y0 - 4.0 <= 0.5 * (d["rect"].y0 + d["rect"].y1)
              <= box.y1 + 2.5]
    # digits are 5.4-5.7 pt tall and upright; the rotated axis title sits
    # further left, and its letters are shorter
    digits = sorted((g for g in glyphs if 5.0 < g.height < 6.2),
                    key=lambda g: -(g.y0 + g.y1))
    rows = []
    for g in digits:
        c = 0.5 * (g.y0 + g.y1)
        if rows and abs(rows[-1][0] - c) < 2.0:
            rows[-1][1].append(g)
        else:
            rows.append([c, [g]])
    return [(sum(0.5 * (g.y0 + g.y1) for g in members) / len(members),
             len(members)) for _, members in rows]


def _scale(rows, labels, zero_y, where):
    """Points per unit and the checks described in the module docstring."""
    if len(rows) != len(labels):
        raise ExtractionError(
            f"{where}: {len(rows)} label rows drawn, {len(labels)} transcribed")
    for (centre, n), text in zip(rows, labels):
        if n != sum(ch.isdigit() for ch in text):
            raise ExtractionError(
                f"{where}: label '{text}' does not have the {n} digits "
                f"drawn at y = {centre:.2f}")
    vals = [float(t) for t in labels]
    ys = [c for c, _ in rows]
    # least-squares line y = a + b * value (b < 0: up the page is smaller y)
    n = len(vals)
    mv, my = sum(vals) / n, sum(ys) / n
    b = (sum((v - mv) * (y - my) for v, y in zip(vals, ys))
         / sum((v - mv) ** 2 for v in vals))
    a = my - b * mv
    misfit = max(abs(a + b * v - y) for v, y in zip(vals, ys))
    if misfit > TOL:
        raise ExtractionError(
            f"{where}: labels {labels} are not evenly spaced in the drawing "
            f"(misfit {misfit:.2f} pt)")
    # the digit box centre sits a fixed 0.42 pt below its tick in this file
    offset = a - zero_y
    if abs(offset - 0.42) > TOL:
        raise ExtractionError(
            f"{where}: the zero implied by the labels is {offset:.2f} pt from "
            "the zero line (expected 0.42)")
    return -b, misfit


def extract(pdf_path: str = PDF) -> list[dict]:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    rows_out = []
    for figure, page_index, shocks in PAGES:
        drawings = doc[page_index].get_drawings()
        panels = []
        for med in _panels(drawings):
            pts = _polyline(med)
            box = _axis_box(drawings, pts)
            panels.append((round(box.y0 / 10.0), round(box.x0, 1), pts, box))
        panels.sort(key=lambda p: (p[0], p[1]))
        for i, (_, _, pts, box) in enumerate(panels):
            shock = shocks[i // 8]
            var = VARIABLES[i % 8]
            where = f"Figure {figure}, {shock} -> {var}"
            ys = _on_horizons(pts, where)
            zero_y = _zero_line(drawings, box)
            labels = TICKS[shock][i % 8]
            per_unit, _ = _scale(_label_rows(drawings, box), labels, zero_y,
                                 where)
            vals = sorted(float(t) for t in labels)
            for h, y in enumerate(ys):
                rows_out.append({
                    "figure": figure,
                    "shock": shock,
                    "variable": var,
                    "unit": UNITS[var],
                    "horizon": h,
                    "median": f"{(zero_y - y) / per_unit:.4f}",
                    "resolution": f"{0.5 * GRID / per_unit:.4f}",
                    "tick_step": f"{vals[1] - vals[0]:g}",
                })
    return rows_out


FIELDS = ["figure", "shock", "variable", "unit", "horizon", "median",
          "resolution", "tick_step"]


def write(rows: list[dict], out: str = OUT) -> None:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    try:
        rows = extract()
    except ImportError:
        sys.exit("PyMuPDF is needed: uv run --with pymupdf python "
                 "scripts/extract_paper_irf_medians.py")
    write(rows, OUT)
    print(f"wrote {len(rows)} rows to {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
