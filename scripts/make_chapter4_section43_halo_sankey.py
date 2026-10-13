#!/usr/bin/env python3
"""
Generate two alternative figures for Chapter 4 Section 4.3.1:
  - fig4_3_1a_concentric_halo.png   (small core, large halo)
  - fig4_3_1b_sankey_flow.png       (governance flow Sankey)

    python3 scripts/make_chapter4_section43_halo_sankey.py
Outputs: figures/chapter4/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path('figures/chapter4')
OUT.mkdir(parents=True, exist_ok=True)

C_GREY   = '#D0D0D0'
C_BLUE   = '#2E75B6'
C_ORANGE = '#E07B3F'
C_DARK   = '#1F3864'
FONT     = 'Arial'

plt.rcParams.update({'font.family': FONT, 'figure.dpi': 300})

# ── Data ──────────────────────────────────────────────────────────────────────
TOTAL     = 47512
EDU_GATE  = 10557   # chunk-gate (22.2%)
TITLE_EDU = 1795    # title-gate (3.8%)
INHERITED = EDU_GATE - TITLE_EDU   # 8,762 (18.4%)
NON_EDU   = TOTAL - EDU_GATE       # 36,955 (77.8%)

PCT_TOTAL    = 100.0
PCT_EDU      = 100 * EDU_GATE  / TOTAL    # 22.2%
PCT_TITLE    = 100 * TITLE_EDU / TOTAL    # 3.8%
PCT_INHERITED = 100 * INHERITED / TOTAL   # 18.4%
PCT_NON      = 100 * NON_EDU   / TOTAL    # 77.8%

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A — Concentric "small core, large halo"
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure A: concentric halo ...")

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.25, 1.25)

# Radii proportional to sqrt(area) so visual area is proportional
R_OUTER = 1.0
R_MID   = R_OUTER * np.sqrt(PCT_EDU   / PCT_TOTAL)   # 0.471
R_INNER = R_OUTER * np.sqrt(PCT_TITLE / PCT_TOTAL)   # 0.195

# Draw rings from outside in
outer = plt.Circle((0, 0), R_OUTER, color=C_GREY,   zorder=1, alpha=0.90)
mid   = plt.Circle((0, 0), R_MID,   color=C_BLUE,   zorder=2, alpha=0.90)
inner = plt.Circle((0, 0), R_INNER, color=C_ORANGE, zorder=3, alpha=0.95)

ax.add_patch(outer)
ax.add_patch(mid)
ax.add_patch(inner)

# Ring outline strokes
for r, c in [(R_OUTER, '#AAAAAA'), (R_MID, '#1A5C9E'), (R_INNER, '#B85A1F')]:
    circle = plt.Circle((0, 0), r, fill=False, edgecolor=c, linewidth=1.5, zorder=10)
    ax.add_patch(circle)

# Inner label
ax.text(0, 0, f'Education-directed\n{TITLE_EDU:,} chunks\n({PCT_TITLE:.1f}%)',
        ha='center', va='center', fontsize=9.5, fontweight='bold',
        color='white', zorder=11, linespacing=1.5)

# Mid-ring label (place in the ring itself)
angle_mid = np.radians(50)
rx = (R_MID + (R_OUTER - R_MID) * 0.3) * np.cos(angle_mid) * (-1)
ry = (R_MID + (R_OUTER - R_MID) * 0.3) * np.sin(angle_mid)
ax.text(rx, ry,
        f'Education-relevant content\nin general-purpose instruments\n{INHERITED:,} chunks ({PCT_INHERITED:.1f}%)',
        ha='center', va='center', fontsize=9, color='white',
        fontweight='bold', zorder=11, linespacing=1.5)

# Outer ring label
angle_out = np.radians(210)
ox = 0.72 * np.cos(angle_out)
oy = 0.72 * np.sin(angle_out)
ax.text(ox, oy,
        f'No education-relevant content\n{NON_EDU:,} chunks ({PCT_NON:.1f}%)',
        ha='center', va='center', fontsize=9, color='#444444',
        fontweight='bold', zorder=11, linespacing=1.5)

# Annotation: the 5:1 ratio
ax.annotate(
    'For every 1 chunk in an\neducation-directed document,\nnearly 5 appear in\ngeneral-purpose instruments',
    xy=(R_INNER * 0.7, R_INNER * 0.7),
    xytext=(0.62, -0.70),
    fontsize=8.5, color=C_DARK, style='italic',
    arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.0,
                    connectionstyle='arc3,rad=-0.25'),
    zorder=12
)

ax.set_title(
    'Education governance in the AI corpus: a small core, a large halo\n'
    'Visual area proportional to chunk count',
    fontsize=12, fontweight='bold', pad=16
)

plt.tight_layout()
out_a = OUT / 'fig4_3_1a_concentric_halo.png'
plt.savefig(out_a, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {out_a.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B — Sankey-style governance flow
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure B: Sankey flow ...")

fig, ax = plt.subplots(figsize=(13, 8))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# ── Helper: draw a filled flow ribbon between two vertical spans ──────────────
def ribbon(ax, x0, x1, y0_bot, y0_top, y1_bot, y1_top, color, alpha=0.35, zorder=1):
    """Cubic bezier ribbon between two vertical extents."""
    xm = (x0 + x1) / 2
    verts = [
        (x0, y0_bot), (xm, y0_bot), (xm, y1_bot), (x1, y1_bot),
        (x1, y1_top), (xm, y1_top), (xm, y0_top), (x0, y0_top),
    ]
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    path = MPath(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha, zorder=zorder)
    ax.add_patch(patch)

def node(ax, x, y_bot, y_top, color, width=0.35, zorder=3):
    rect = mpatches.FancyBboxPatch(
        (x - width/2, y_bot), width, y_top - y_bot,
        boxstyle='round,pad=0.03',
        facecolor=color, edgecolor='white', linewidth=1.5, zorder=zorder
    )
    ax.add_patch(rect)

# ── Layout ────────────────────────────────────────────────────────────────────
# X positions: source | split1 | split2
X0, X1, X2 = 1.2, 4.5, 8.0
NODE_W = 0.4

# Total height = 9 units (1–9). Scale: 1 unit = 100% / 9
SCALE = 8.0 / 100   # units per percent

# Source node: full corpus
src_bot, src_top = 1.0, 9.0
node(ax, X0, src_bot, src_top, C_GREY, width=NODE_W)
ax.text(X0, (src_bot+src_top)/2,
        f'Full corpus\n{TOTAL:,} chunks\n(100%)',
        ha='center', va='center', fontsize=9.5, fontweight='bold',
        color='#333333', zorder=10)

# First split: edu-relevant vs non-edu
# edu-relevant = 22.2%  → top portion
# non-edu      = 77.8%  → bottom portion
edu_height = PCT_EDU   * SCALE   # 1.78
non_height = PCT_NON   * SCALE   # 6.22

edu_bot1  = src_top - edu_height   # 7.22
edu_top1  = src_top                # 9.0
non_bot1  = src_bot                # 1.0
non_top1  = src_bot + non_height   # 7.22  (same boundary)

node(ax, X1, edu_bot1, edu_top1, C_BLUE,   width=NODE_W)
node(ax, X1, non_bot1, non_top1, C_GREY,   width=NODE_W)

ax.text(X1, (edu_bot1+edu_top1)/2,
        f'Education-relevant\n{EDU_GATE:,} chunks\n({PCT_EDU:.1f}%)',
        ha='center', va='center', fontsize=8.5, fontweight='bold',
        color='white', zorder=10)
ax.text(X1, (non_bot1+non_top1)/2,
        f'Not education-relevant\n{NON_EDU:,} chunks\n({PCT_NON:.1f}%)',
        ha='center', va='center', fontsize=8.5, fontweight='bold',
        color='#444444', zorder=10)

# Ribbons source → split1
ribbon(ax, X0+NODE_W/2, X1-NODE_W/2,
       edu_bot1, edu_top1, edu_bot1, edu_top1,
       color=C_BLUE, alpha=0.25)
ribbon(ax, X0+NODE_W/2, X1-NODE_W/2,
       non_bot1, non_top1, non_bot1, non_top1,
       color=C_GREY, alpha=0.20)

# Second split: inherited vs direct (within the edu-relevant flow)
title_height = PCT_TITLE    * SCALE   # 0.304
inh_height   = PCT_INHERITED * SCALE  # 1.474

dir_bot2 = edu_top1 - title_height   # 8.696
dir_top2 = edu_top1                  # 9.0
inh_bot2 = edu_bot1                  # 7.22
inh_top2 = edu_bot1 + inh_height     # 8.696

node(ax, X2, dir_bot2, dir_top2, C_ORANGE, width=NODE_W)
node(ax, X2, inh_bot2, inh_top2, C_BLUE,   width=NODE_W)

ax.text(X2, (dir_bot2+dir_top2)/2,
        f'Education-directed\n{TITLE_EDU:,} chunks\n({PCT_TITLE:.1f}%)',
        ha='center', va='center', fontsize=8, fontweight='bold',
        color='white', zorder=10)
ax.text(X2, (inh_bot2+inh_top2)/2,
        f'General-purpose instruments\nreaching education\n{INHERITED:,} chunks ({PCT_INHERITED:.1f}%)',
        ha='center', va='center', fontsize=8, fontweight='bold',
        color='white', zorder=10)

# Ribbons split1 → split2 (edu flow only)
ribbon(ax, X1+NODE_W/2, X2-NODE_W/2,
       dir_bot2, dir_top2, dir_bot2, dir_top2,
       color=C_ORANGE, alpha=0.30)
ribbon(ax, X1+NODE_W/2, X2-NODE_W/2,
       inh_bot2, inh_top2, inh_bot2, inh_top2,
       color=C_BLUE, alpha=0.25)

# ── Domain labels on the right ────────────────────────────────────────────────
# Inherited: governance domains
inh_domains = [
    'Data protection & privacy',
    'Risk classification',
    'Liability & accountability',
    'Workforce & skills policy',
    'Digital transformation',
    'Public-sector modernisation',
    'Research & innovation policy',
    'Procurement & programme implementation',
]
# Direct: document types
dir_domains = [
    'AI literacy frameworks',
    'Generative AI in schools guidance',
    'Digital education strategies',
    'Higher education research policy',
]

# Inherited domain labels
inh_mid = (inh_bot2 + inh_top2) / 2
for i, d in enumerate(inh_domains):
    y = inh_top2 - 0.06 - i * (inh_height / (len(inh_domains) + 0.5))
    ax.text(X2 + NODE_W/2 + 0.15, y, f'· {d}',
            ha='left', va='center', fontsize=7.5, color='#2E75B6')

# Direct domain labels
for i, d in enumerate(dir_domains):
    y = dir_top2 - 0.06 - i * (title_height / (len(dir_domains) + 0.5))
    ax.text(X2 + NODE_W/2 + 0.15, y, f'· {d}',
            ha='left', va='center', fontsize=7.5, color='#B85A1F')

# ── Column headers ────────────────────────────────────────────────────────────
for x, label in [(X0, 'Full corpus'), (X1, 'Education filter'), (X2, 'Instrument type')]:
    ax.text(x, 9.45, label, ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color='#1F3864')

# ── Source note ───────────────────────────────────────────────────────────────
ax.text(0.5, 0.3,
        'Source: chunk-level education gate and title-level education gate (CIAP pipeline).\n'
        'Flow width proportional to chunk count.',
        ha='left', va='bottom', fontsize=7.5, color='#888888', style='italic')

ax.set_title(
    'How education governance reaches the AI corpus: governance flow by instrument type',
    fontsize=12, fontweight='bold', pad=14
)

plt.tight_layout()
out_b = OUT / 'fig4_3_1b_sankey_flow.png'
plt.savefig(out_b, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {out_b.name}")

print()
print("Done. Figures saved to:", OUT)
for f in sorted(OUT.iterdir()):
    if f.suffix == '.png' and 'fig4_3_1' in f.name:
        print(f"  {f.name:55s}  {f.stat().st_size/1024:6.1f} KB")
