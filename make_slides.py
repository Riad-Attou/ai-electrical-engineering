#!/usr/bin/env python3
"""Build a native, editable PowerPoint for the AI in Electrical Engineering deck.

Content: Servo backlash compensation with deep learning (ML2).
Design: warm paper background, left rule, Georgia / Calibri / Consolas.

    python make_slides.py   ->   presentation.pptx
"""
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG  = HERE / "figures"
OUT  = HERE / "presentation.pptx"

# ── palette ────────────────────────────────────────────────────────────────
PAPER  = RGBColor(0xFB, 0xFA, 0xF6)
PAPER2 = RGBColor(0xF4, 0xF1, 0xE9)
INK    = RGBColor(0x1C, 0x2B, 0x36)
INKSOFT= RGBColor(0x4A, 0x5A, 0x66)
RULE   = RGBColor(0xD9, 0xD2, 0xC4)
NAVY   = RGBColor(0x1D, 0x35, 0x57)
TEAL   = RGBColor(0x2A, 0x9D, 0x8F)
TERRA  = RGBColor(0xE0, 0x7A, 0x5F)
GOLD   = RGBColor(0xC9, 0x9A, 0x2E)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
TEAL_TINT  = RGBColor(0xE7, 0xF2, 0xF0)
TERRA_TINT = RGBColor(0xFA, 0xEC, 0xE7)
GOLD_TINT  = RGBColor(0xFB, 0xF3, 0xDE)

SERIF, SANS, MONO = "Georgia", "Calibri", "Consolas"
ACCENT = {"teal": TEAL, "navy": NAVY, "terra": TERRA, "gold": GOLD}

SW, SH = Inches(13.333), Inches(7.5)
MX     = Inches(0.95)
RULE_X = Inches(0.62)
CW     = Inches(11.6)

prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


# ── low-level helpers ──────────────────────────────────────────────────────

def slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = PAPER
    rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, RULE_X, Inches(0.55), Pt(1.4), Inches(6.4))
    _flat(rule, RULE)
    cap = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, RULE_X, Inches(0.55), Pt(2.4), Inches(0.62))
    _flat(cap, TEAL)
    return s


def _flat(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def box(s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    tf.paragraphs[0].text = ""
    return tf


_TOKEN = re.compile(r"\*([^*]+)\*|~([^~]+)~|`([^`]+)`|([^*~`]+)")


def runs(p, text, font=SANS, color=INKSOFT, size=15, bold=False):
    for m in _TOKEN.finditer(text):
        strong, em, mono, plain = m.groups()
        r = p.add_run()
        if strong is not None:
            r.text, r.font.name, r.font.bold, r.font.color.rgb = strong, font, True, INK
            r.font.size = Pt(size)
        elif em is not None:
            r.text, r.font.name, r.font.bold, r.font.color.rgb = em, font, True, TEAL
            r.font.size = Pt(size)
        elif mono is not None:
            r.text, r.font.name, r.font.bold, r.font.color.rgb = mono, MONO, False, NAVY
            r.font.size = Pt(size * 0.92)
        else:
            r.text, r.font.name, r.font.bold, r.font.color.rgb = plain, font, bold, color
            r.font.size = Pt(size)


def para(tf, text, *, font=SANS, color=INKSOFT, size=15, bold=False,
         align=PP_ALIGN.LEFT, before=0, after=6, lh=1.12, first=False):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align
    p.space_before, p.space_after, p.line_spacing = Pt(before), Pt(after), lh
    runs(p, text, font=font, color=color, size=size, bold=bold)
    return p


def kicker(s, num, label, top=Inches(0.6)):
    tf = box(s, MX, top, CW, Inches(0.35))
    p = tf.paragraphs[0]; p.line_spacing = 1.0
    if num:
        r = p.add_run(); r.text = num + "  "
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = MONO, Pt(11), True, TERRA
    r = p.add_run(); r.text = label.upper()
    r.font.name, r.font.size, r.font.color.rgb = MONO, Pt(11), TEAL


def heading(s, text, top, size=30, w=CW):
    tf = box(s, MX, top, w, Inches(1.4))
    para(tf, text, font=SERIF, color=NAVY, size=size, bold=True, lh=1.02, first=True)


def subhead(s, text, l, t, w):
    tf = box(s, l, t, w, Inches(0.5))
    para(tf, text, font=SERIF, color=NAVY, size=19, bold=True, lh=1.05, first=True)


def bullets(s, items, l, t, w, h, size=15, gap=10):
    tf = box(s, l, t, w, h)
    for i, (txt, accent) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before, p.space_after, p.line_spacing = Pt(0 if i == 0 else gap), Pt(0), 1.1
        d = p.add_run(); d.text = "—  "
        d.font.name, d.font.size, d.font.bold = SANS, Pt(size), True
        d.font.color.rgb = ACCENT.get(accent, TERRA)
        runs(p, txt, font=SANS, color=INKSOFT, size=size)
    return tf


def card(s, l, t, w, h, *, fill=WHITE, accent=None, line=RULE):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    c.adjustments[0] = 0.06
    c.fill.solid(); c.fill.fore_color.rgb = fill
    c.line.color.rgb = line; c.line.width = Pt(1)
    c.shadow.inherit = False
    if accent:
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, Pt(3), h)
        _flat(bar, ACCENT[accent])
    return c


def formula(s, lines, l, t, w, *, accent="terra", size=13.5, h=None):
    if h is None:
        h = Inches(0.38) + Inches(0.32) * len(lines)
    card(s, l, t, w, h, fill=PAPER2, accent=accent)
    tf = box(s, l + Inches(0.22), t + Inches(0.12), w - Inches(0.4), h - Inches(0.2),
             anchor=MSO_ANCHOR.MIDDLE)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before, p.space_after, p.line_spacing = Pt(0 if i == 0 else 4), Pt(0), 1.15
        runs(p, ln, font=MONO, color=NAVY, size=size)


def takeaway(s, bold, text, l, t, w, *, accent="teal", h=Inches(0.95)):
    tint = TEAL_TINT if accent == "teal" else (TERRA_TINT if accent == "terra" else GOLD_TINT)
    card(s, l, t, w, h, fill=tint, accent=accent, line=tint)
    tf = box(s, l + Inches(0.22), t + Inches(0.1), w - Inches(0.42), h - Inches(0.2),
             anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]; p.line_spacing = 1.12
    r = p.add_run(); r.text = bold + "  "
    r.font.name, r.font.bold, r.font.size, r.font.color.rgb = SERIF, True, Pt(15), NAVY
    runs(p, text, font=SANS, color=INKSOFT, size=15)


def pills(s, items, t, *, h=Inches(1.9)):
    n = len(items)
    gap = Inches(0.25)
    w = (CW - gap * (n - 1)) / n
    for i, (tag, tac, title, body) in enumerate(items):
        l = MX + (w + gap) * i
        card(s, l, t, w, h, accent=tac)
        tf = box(s, l + Inches(0.22), t + Inches(0.18), w - Inches(0.4), h - Inches(0.3))
        para(tf, tag.upper(), font=MONO, color=ACCENT[tac], size=10, first=True, after=4)
        para(tf, title, font=SERIF, color=NAVY, size=16, bold=True, after=6, lh=1.0)
        para(tf, body, font=SANS, color=INKSOFT, size=12, lh=1.1)


def roadmap(s, items, t):
    gap = Inches(0.22)
    w = (CW - gap) / 2
    h = Inches(1.5)
    for i, (rn, title, body) in enumerate(items):
        r, c = divmod(i, 2)
        l = MX + (w + gap) * c
        tt = t + (h + gap) * r
        card(s, l, tt, w, h)
        nb = box(s, l + Inches(0.2), tt + Inches(0.16), Inches(0.7), Inches(1.2))
        para(nb, rn, font=SERIF, color=TERRA, size=30, bold=True, first=True, lh=1.0)
        tf = box(s, l + Inches(0.95), tt + Inches(0.16), w - Inches(1.15), h - Inches(0.3))
        para(tf, title, font=SERIF, color=NAVY, size=17, bold=True, first=True, after=4, lh=1.0)
        para(tf, body, font=SANS, color=INKSOFT, size=12, lh=1.12)


def stats(s, items, t):
    gap = Inches(0.3)
    w = (CW - gap * 2) / 3
    for i, (val, ck, label) in enumerate(items):
        l = MX + (w + gap) * i
        top = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, Pt(2))
        _flat(top, RULE)
        tf = box(s, l, t + Inches(0.16), w, Inches(1.7))
        para(tf, val, font=SERIF, color=ACCENT.get(ck, NAVY), size=44, bold=True, first=True, lh=1.0)
        para(tf, label.upper(), font=MONO, color=INKSOFT, size=10, before=8, lh=1.1)


def figure(s, name, caption, l, t, w):
    path = FIG / name
    iw, ih = Image.open(path).size
    pad = Inches(0.16)
    disp_w = w - pad * 2
    disp_h = int(disp_w * ih / iw)
    max_h = Inches(4.35)
    if disp_h > max_h:
        disp_h = max_h
        disp_w = int(disp_h * iw / ih)
    cap_h = Inches(0.5)
    card_h = disp_h + pad * 2 + cap_h
    card(s, l, t, w, card_h)
    img_l = l + (w - disp_w) // 2
    s.shapes.add_picture(str(path), img_l, t + pad, width=disp_w, height=disp_h)
    tf = box(s, l + pad, t + pad + disp_h + Inches(0.05), w - pad * 2, cap_h)
    para(tf, caption, font=MONO, color=INKSOFT, size=11, first=True, lh=1.1)


def tbl(s, rows, l, t, w):
    n = len(rows)
    gt = s.shapes.add_table(n, 2, l, t, w, Inches(0.45) * n).table
    gt.columns[0].width = int(w * 0.42)
    gt.columns[1].width = w - int(w * 0.42)
    gt.first_row = False; gt.horz_banding = False
    for ri, (k, v) in enumerate(rows):
        for ci, (txt, fnt, col, sz) in enumerate(
            [(k, MONO, NAVY, 12), (v, SANS, INKSOFT, 13)]
        ):
            cell = gt.cell(ri, ci)
            cell.fill.solid(); cell.fill.fore_color.rgb = PAPER
            cell.margin_left = Inches(0.08); cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            runs(p, txt, font=fnt, color=col, size=sz)


# ── placeholder slide (clean — no boxes, no guidance) ─────────────────────

def placeholder(num, kick, title):
    s = slide()
    kicker(s, num, kick)
    heading(s, title, Inches(1.05), size=30)
    return s


# ── slides ─────────────────────────────────────────────────────────────────

def s01_title():
    s = slide()
    kicker(s, "", "AI in Electrical Engineering", top=Inches(0.9))
    tf = box(s, MX, Inches(1.4), Inches(8.3), Inches(2.0))
    para(tf, "[Presentation title]", font=SERIF, color=NAVY, size=38, bold=True, first=True, lh=1.04)
    tf2 = box(s, MX, Inches(3.3), Inches(7.5), Inches(0.7))
    p = tf2.paragraphs[0]; p.line_spacing = 1.2
    runs(p, "[Short subtitle — one sentence describing the project]",
         font=SANS, color=INKSOFT, size=18)
    meta = [("[Author 1]  ·  [Author 2]", "[Author 3]  ·  [Author 4]"),
            ("Course", "AI in Electrical Engineering"),
            ("[University]", "[Year]")]
    mw = Inches(3.0)
    for i, (a, b) in enumerate(meta):
        tf = box(s, MX + mw * i, Inches(4.45), mw - Inches(0.2), Inches(0.8))
        para(tf, a, font=MONO, color=NAVY, size=12, bold=True, first=True, after=2)
        para(tf, b, font=MONO, color=INKSOFT, size=11, lh=1.1)


def s02_toc():
    s = slide()
    kicker(s, "", "Table of contents")
    heading(s, "Outline.", Inches(1.02), size=28)
    roadmap(s, [
        ("I",   "Physical setup & dataset",
         "Two-inertia servo ODE, chirp training excitation, multisine test."),
        ("II",  "ML1 — Motor identification",
         "[Friend's section] Sparse identification of motor dynamics."),
        ("III", "ML2 — Backlash compensation",
         "GRU, CNN, TCN: per-window normalisation, architectures, results."),
        ("IV",  "Conclusion",
         "Key numbers, takeaways, and future directions."),
    ], Inches(2.0))


def s03_ph_context():
    placeholder("01", "Physical setup & dataset", "[Physical setup & dataset]")


def s04_ph_ml1a():
    placeholder("02", "ML1 — motor identification", "[ML1 — motor identification]")


def s04b_ph_ml1b():
    placeholder("02", "ML1 — results", "[ML1 — results & validation]")


def s05_normalization():
    s = slide()
    kicker(s, "03", "ML2 — feature engineering")
    heading(s, "Per-window velocity normalisation.", Inches(1.05), size=28)
    tf = box(s, MX, Inches(2.0), CW, Inches(0.72))
    p = tf.paragraphs[0]; p.line_spacing = 1.2
    runs(p, "Backlash error ≈ damper lag ∝ motor velocity. The chirp (train) is 2.4× faster than the multisine (test). A fixed `error_std` from training causes 2.3× over-correction on test — all models score *worse than rigid coupling*.",
         font=SANS, color=INKSOFT, size=15)
    formula(s, [
        "local_vel  =  std( diff(enc_m_window) )          ← velocity proxy per window",
        "enc_m_rel[t]  =  (enc_m[t] − enc_m[t₀]) / local_vel   ← input feature",
        "y_norm  =  (θ_l − enc_m/N) / (local_vel / N)            ← target",
        "pred_error  =  model_output × (local_vel / N)            ← denorm at inference",
    ], MX, Inches(2.88), CW, accent="navy", h=Inches(1.6))
    bullets(s, [
        ("Both input and target scale with the *same local speed* — the ratio is constant across train and test.", "teal"),
        ("After the fix, normalised-target std ratio train vs test: *1.06×*  (was 2.3×).", "navy"),
        ("Window W = 64 steps = 64 ms provides enough velocity history without excessive delay.", "terra"),
    ], MX, Inches(4.72), CW, Inches(2.1), size=15, gap=12)


def s06_architectures():
    s = slide()
    kicker(s, "04", "ML2 — model architectures")
    heading(s, "Three architectures, three context lengths.", Inches(1.05), size=28)
    pills(s, [
        ("gru", "teal",  "GRU  (3 489 params)",
         "Recurrent hidden state h accumulates *64 ms* of context. hidden = 32, 1 layer."),
        ("cnn", "navy",  "CNN  (8 801 params)",
         "Stack of valid 1-D convolutions. Receptive field = *15 ms*. channels = 32, k = 8, depth = 2."),
        ("tcn", "terra", "TCN  (33 153 params)",
         "Dilated causal convolutions. Receptive field = *91 ms*. channels = 32, k = 4, 4 levels."),
    ], Inches(2.55), h=Inches(2.05))
    tf = box(s, MX, Inches(5.12), CW, Inches(0.55))
    p = tf.paragraphs[0]; p.line_spacing = 1.2
    runs(p, "All models: input `(B, 64, 2)` — [enc_m_rel, pwm_norm] — output scalar per-window normalised error. Loss: MSE. Optimiser: Adam lr = 1e-3.",
         font=SANS, color=INKSOFT, size=14)
    formula(s, [
        "RF_CNN  =  (k−1) × depth × 2 + 1  =  15 steps",
        "RF_TCN  =  (k−1) × (2^levels − 1) × 2 + 1  =  91 steps",
    ], MX, Inches(5.88), CW, accent="terra", h=Inches(0.88))


def s07_results():
    s = slide()
    kicker(s, "05", "ML2 — results  (test set, 30 s multisine)")
    heading(s, "TCN +48 %, GRU +29 % vs rigid coupling.", Inches(1.05), size=28)
    tbl(s, [
        ("Rigid coupling  —  enc_m / N",   "RMSE  0.102 mrad   (0.0058°)    —   reference"),
        ("Output encoder  —  enc_o",        "RMSE  0.323 mrad   (0.0185°)   −217 %"),
        ("CNN  —  valid conv, RF = 15 ms",  "RMSE  0.117 mrad   (0.0067°)   −14.8 %"),
        ("*GRU*  —  W = 64 ms, hidden 32",  "RMSE  0.072 mrad   (0.0041°)   *+29.4 %*"),
        ("*TCN 🏆*  —  dilated, RF = 91 ms","RMSE  0.052 mrad   (0.0030°)   *+48.4 %*"),
    ], MX, Inches(2.2), CW)
    takeaway(s, "Why CNN fails.", "Its 15 ms receptive field cannot estimate velocity from a *noisy quantised encoder*. GRU (64 ms) and TCN (91 ms) are long enough to smooth the quantisation noise and measure velocity accurately.",
             MX, Inches(4.95), CW, accent="terra")
    tf = box(s, MX, Inches(6.15), CW, Inches(0.8))
    p = tf.paragraphs[0]; p.line_spacing = 1.2
    runs(p, "Error is *damper-induced lag* proportional to motor velocity. A longer context gives a better velocity estimate — which directly explains the receptive-field ordering.",
         font=SANS, color=INKSOFT, size=14)


def s08_fig_comparison():
    s = slide()
    kicker(s, "06", "ML2 — comparison figure  (zoom 2–5 s)")
    heading(s, "TCN tracks the true backlash error closely.", Inches(1.05), size=27)
    bullets(s, [
        ("True backlash error (dashed) stays within *±0.6 mrad* — damper lag, not dead-zone hysteresis.", "navy"),
        ("TCN *follows the oscillation*; GRU close behind; CNN adds noise instead of removing it.", "teal"),
        ("Rigid coupling error oscillates in phase with motor velocity — confirming the *v-proportional* mechanism.", "terra"),
    ], MX, Inches(2.2), Inches(4.5), Inches(2.5), size=15, gap=13)
    figure(s, "comparison_backlash_zoom.png",
           "Fig. 1 — backlash residual and estimation error, zoom 2–5 s (test set, multisine).",
           Inches(5.7), Inches(1.35), Inches(6.7))


def s09_analysis():
    s = slide()
    kicker(s, "07", "ML2 — analysis")
    heading(s, "The gear spring never fires.", Inches(1.05), size=30)
    tf = box(s, MX, Inches(2.0), Inches(6.2), Inches(0.88))
    p = tf.paragraphs[0]; p.line_spacing = 1.2
    runs(p, "In the simulation, `T_mesh = kg·dz + cg·(ω_m − N·ω_l)`. The spring `kg·dz` fires only when |φ| > gap_motor = 1 rad. But φ = θ_m − N·θ_l stays within *±0.03 rad* throughout all runs.",
         font=SANS, color=INKSOFT, size=15)
    rl = Inches(7.6)
    formula(s, [
        "φ  =  θ_m − N·θ_l  ≤  0.03 rad   «   gap_motor = 1.0 rad",
        "→  dz = 0   →   T_mesh = cg·(ω_m − N·ω_l)   always",
        "τ_l = Jl / (N²·η·cg)  =  5e-3 / 42.5  ≈  0.12 ms",
    ], rl, Inches(2.0), Inches(4.7), accent="terra")
    bullets(s, [
        ("The task is *predicting damper lag* — proportional to velocity — not dead-zone hysteresis.", "terra"),
        ("*Longer context* improves the velocity estimate from noisy quantised encoder data.", "navy"),
        ("TCN (91 ms) > GRU (64 ms) > CNN (15 ms) — result follows directly from context length.", "teal"),
    ], MX, Inches(3.4), Inches(6.0), Inches(2.7), size=15, gap=12)
    takeaway(s, "Next step.",
             "Reduce the output gap to 0.001 rad so the spring activates — that would stress-test *hysteresis learning* and likely widen the gap between GRU and TCN.",
             MX, Inches(6.35), CW)


def s10_conclusion():
    placeholder("08", "Conclusion", "[Conclusion]")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    for fn in [
        s01_title,
        s02_toc,
        s03_ph_context,
        s04_ph_ml1a,
        s04b_ph_ml1b,
        s05_normalization,
        s06_architectures,
        s07_results,
        s08_fig_comparison,
        s09_analysis,
        s10_conclusion,
    ]:
        fn()
    prs.save(OUT)
    print(f"Wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
