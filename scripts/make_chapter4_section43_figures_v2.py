#!/usr/bin/env python3
"""
Fixed figures for Chapter 4, Sections 4.3.1 and 4.3.2.
Run from project root with venv active:
    python3 scripts/make_chapter4_section43_figures_v2.py
Outputs: figures/chapter4/
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

OUT = Path('figures/chapter4')
OUT.mkdir(parents=True, exist_ok=True)

C_BLUE   = '#2E75B6'
C_ORANGE = '#E07B3F'
C_GREY   = '#BDBDBD'
C_DARK   = '#1F3864'
C_LIGHT  = '#D6E8F7'
FONT     = 'Arial'

plt.rcParams.update({
    'font.family': FONT,
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

STEP9  = Path('data/derived/step9_country_dataset')
STEP10 = Path('data/derived/step10_education_dataset')

overall      = pd.read_csv(STEP9  / 'country_governance_dataset.csv')
edu          = pd.read_csv(STEP10 / 'education_country_dataset.csv')
clusters_edu = pd.read_csv(STEP10 / 'education_cluster_assignments.csv')
contrastive  = pd.read_csv(
    'data/derived/step8_analysis_title_clean/contrastive_table_full_vs_edu_vs_title.csv'
)

print("Data loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Waffle chart (corrected title and labels)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 1: waffle chart ...")

total     = 47512
edu_total = 10557
title_edu = 1795
inherited = edu_total - title_edu   # 8,762
non_edu   = total - edu_total        # 36,955

# Each of 100 cells ≈ 1% of total chunks
n_non  = round(non_edu   / total * 100)   # 78
n_inh  = round(inherited / total * 100)   # 18
n_dir  = 100 - n_non - n_inh              # 4

COLS, ROWS = 20, 5
colors_map = (
    [C_GREY]   * n_non +
    [C_BLUE]   * n_inh +
    [C_ORANGE] * n_dir
)

fig, ax = plt.subplots(figsize=(11, 5))
ax.set_aspect('equal')
ax.axis('off')

for i, color in enumerate(colors_map):
    r = i // COLS
    c = i  % COLS
    rect = mpatches.FancyBboxPatch(
        (c * 1.25, (ROWS - 1 - r) * 1.25),
        1.05, 1.05,
        boxstyle="round,pad=0.06",
        facecolor=color,
        edgecolor='white',
        linewidth=1.8,
    )
    ax.add_patch(rect)

ax.set_xlim(-0.3, COLS * 1.25 + 0.3)
ax.set_ylim(-2.5, ROWS * 1.25 + 2.2)

legend_items = [
    mpatches.Patch(facecolor=C_GREY,   edgecolor='white',
                   label=f'No education-relevant content   ({non_edu:,} chunks, 77.8%)'),
    mpatches.Patch(facecolor=C_BLUE,   edgecolor='white',
                   label=f'Education-relevant content in general-purpose instruments   ({inherited:,} chunks, 18.4%)'),
    mpatches.Patch(facecolor=C_ORANGE, edgecolor='white',
                   label=f'Education-relevant content in education-directed instruments   ({title_edu:,} chunks, 3.8%)'),
]
ax.legend(
    handles=legend_items,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=1,
    frameon=False,
    fontsize=9.5,
)

ax.set_title(
    'How education-relevant governance reaches the corpus\n'
    'Each cell ≈ 1% of 47,512 paragraph-based chunks',
    fontsize=12, fontweight='bold', pad=14,
)

plt.tight_layout()
out1 = OUT / 'fig4_3_1_corpus_split_waffle_v2.png'
plt.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {out1.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Statement type comparison (arrow removed, annotation added)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 2: statement types ...")

corpora   = ['Full corpus\n(874 docs)', 'Education-relevant\n(671 docs)', 'Education-directed\n(34 docs)']
strategy  = [40.13, 42.66, 52.89]
norm      = [54.14, 51.28, 41.49]
rule      = [1.70,  2.23,  1.15]
other     = [4.03,  3.83,  4.47]

x, w = np.arange(len(corpora)), 0.18

fig, ax = plt.subplots(figsize=(10, 6.5))

b1 = ax.bar(x - 1.5*w, strategy, w, label='Strategy',                color=C_GREY,  alpha=0.92)
b2 = ax.bar(x - 0.5*w, norm,     w, label='Norm',                    color=C_BLUE,  alpha=0.92)
b3 = ax.bar(x + 0.5*w, rule,     w, label='Rule',                    color=C_DARK,  alpha=0.92)
b4 = ax.bar(x + 1.5*w, other,    w, label='Other / low confidence',  color=C_LIGHT,
            edgecolor=C_GREY, linewidth=0.8, alpha=0.92)

for bars, vals in [(b1, strategy), (b2, norm), (b3, rule)]:
    for bar, v in zip(bars, vals):
        if v > 0.8:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

# Annotate the strategy trend with a bracket instead of an arrow
for i, (xi, s) in enumerate(zip(x, strategy)):
    offset = -1.5*w
    if i > 0:
        ax.annotate(
            '', xy=(xi + offset, s + 1.5),
            xytext=(x[i-1] + offset, strategy[i-1] + 1.5),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2),
        )
ax.text(x[1] - 1.5*w, max(strategy) + 5,
        'Strategy share rises\nacross corpora', ha='center',
        fontsize=8.5, color='#666666', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(corpora, fontsize=10)
ax.set_ylabel('Share of institutional statements (%)', fontsize=10)
ax.set_title(
    'Statement type distribution across three analytical corpora',
    fontsize=12, fontweight='bold',
)
ax.legend(frameon=False, fontsize=9.5, loc='upper right')
ax.set_ylim(0, 70)

plt.tight_layout()
out2 = OUT / 'fig4_3_4_statement_types_three_corpora_v2.png'
plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {out2.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Rank shift ribbons (fixed layout)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 3: rank shift ribbons ...")

# Build rank tables
overall_r = (overall[['country','n_statements']]
             .sort_values('n_statements', ascending=False)
             .reset_index(drop=True))
overall_r['overall_rank'] = overall_r.index + 1

edu_r = (edu[['country','n_statements']]
         .sort_values('n_statements', ascending=False)
         .reset_index(drop=True))
edu_r['edu_rank'] = edu_r.index + 1
edu_r = edu_r.rename(columns={'n_statements': 'edu_statements'})

merged = overall_r.merge(edu_r, on='country', how='inner')

# Show top 20 from EACH side (union)
left_top  = set(merged.sort_values('overall_rank').head(20)['country'])
right_top = set(merged.sort_values('edu_rank').head(20)['country'])
show_countries = left_top | right_top  # union — all that appear on either side

df = merged[merged['country'].isin(show_countries)].copy()
df = df.sort_values('overall_rank')

# Assign y-positions: left side = overall rank order (1-N, top=1)
# right side = edu rank order
left_countries  = df.sort_values('overall_rank')['country'].tolist()
right_countries = df.sort_values('edu_rank')['country'].tolist()

# Only keep countries that appear in top-20 of their respective side
left_20  = merged.sort_values('overall_rank').head(20)['country'].tolist()
right_20 = merged.sort_values('edu_rank').head(20)['country'].tolist()

fig, ax = plt.subplots(figsize=(11, 10))
ax.axis('off')

N_LEFT  = len(left_20)
N_RIGHT = len(right_20)

Y_STEP = 1.0
LEFT_X  = 0.0
RIGHT_X = 1.0
DOT_R   = 0.018

def y_pos(rank, n):
    return (n - rank) * Y_STEP

# Draw ribbons
for _, row in merged.iterrows():
    country = row['country']
    on_left  = country in left_20
    on_right = country in right_20
    if not (on_left or on_right):
        continue

    rank_change = row['overall_rank'] - row['edu_rank']
    if rank_change > 2:
        color, alpha, lw = C_ORANGE, 0.85, 1.8   # rose in education layer
    elif rank_change < -2:
        color, alpha, lw = C_BLUE,   0.60, 1.4   # fell in education layer
    else:
        color, alpha, lw = '#999999', 0.45, 1.0  # stable

    if on_left and on_right:
        y_left  = y_pos(left_20.index(country)  + 1, N_LEFT)
        y_right = y_pos(right_20.index(country) + 1, N_RIGHT)
        ax.plot([LEFT_X, RIGHT_X], [y_left, y_right],
                color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')

# Draw dots and labels
for i, country in enumerate(left_20):
    y = y_pos(i + 1, N_LEFT)
    ax.plot(LEFT_X, y, 'o', color='#444444', markersize=6, zorder=5)
    ax.text(LEFT_X - 0.03, y, f'{i+1}. {country}',
            ha='right', va='center', fontsize=8.5)

for i, country in enumerate(right_20):
    row   = merged[merged['country'] == country].iloc[0]
    y     = y_pos(i + 1, N_RIGHT)
    stmts = int(row['edu_statements'])
    ax.plot(RIGHT_X, y, 'o', color='#444444', markersize=6, zorder=5)
    ax.text(RIGHT_X + 0.03, y, f'{i+1}. {country} ({stmts:,})',
            ha='left', va='center', fontsize=8.5)

# Column headers
top_y = Y_STEP * max(N_LEFT, N_RIGHT) * 0.5 + Y_STEP * 0.8
ax.text(LEFT_X,  top_y, 'Overall AI governance rank',
        ha='center', va='bottom', fontsize=10.5, fontweight='bold')
ax.text(RIGHT_X, top_y, 'Education-relevant rank\n(statements)',
        ha='center', va='bottom', fontsize=10.5, fontweight='bold')

# Legend
legend_items = [
    mlines.Line2D([],[], color=C_ORANGE, lw=2.0,
                  label='Rose 3+ positions in education layer'),
    mlines.Line2D([],[], color=C_BLUE,   lw=1.8,
                  label='Fell 3+ positions in education layer'),
    mlines.Line2D([],[], color='#999999', lw=1.2,
                  label='Broadly stable (within 2 positions)'),
]
ax.legend(handles=legend_items, loc='lower center',
          bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9, ncol=3)

ax.set_title('Jurisdictional rank shifts: overall AI governance vs education-relevant governance',
             fontsize=12, fontweight='bold', pad=14)

ax.set_xlim(-0.55, 1.55)
ax.set_ylim(-1.5, top_y + 0.5)

plt.tight_layout()
out3 = OUT / 'fig4_3_2b_rank_shift_ribbons_v2.png'
plt.savefig(out3, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {out3.name}")

print()
print("All figures saved to:", OUT)
for f in sorted(OUT.iterdir()):
    if f.suffix == '.png':
        print(f"  {f.name:60s}  {f.stat().st_size/1024:7.1f} KB")
