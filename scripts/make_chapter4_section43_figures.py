#!/usr/bin/env python3
"""
Generate all figures for dissertation Chapter 4, Sections 4.3.1 and 4.3.2.
Run from the project root: python3 scripts/make_chapter4_section43_figures.py
Outputs go to: figures/chapter4/
"""
 
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
 
# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path('figures/chapter4')
OUT.mkdir(parents=True, exist_ok=True)
 
# ── Colour palette (dissertation-consistent) ──────────────────────────────────
C_BLUE    = '#2E75B6'
C_ORANGE  = '#E07B3F'
C_GREEN   = '#4CAF50'
C_GREY    = '#BDBDBD'
C_DARK    = '#1F3864'
C_LIGHT   = '#EBF3FB'
FONT      = 'Arial'
 
plt.rcParams.update({
    'font.family': FONT,
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})
 
# ── Load data ─────────────────────────────────────────────────────────────────
STEP9 = Path('data/derived/step9_country_dataset')
STEP10 = Path('data/derived/step10_education_dataset')
 
overall      = pd.read_csv(STEP9 / 'country_governance_dataset.csv')
edu          = pd.read_csv(STEP10 / 'education_country_dataset.csv')
clusters_edu = pd.read_csv(STEP10 / 'education_cluster_assignments.csv')
 
# Merge edu with clusters
edu = edu.merge(clusters_edu[['country','cluster_id']], on='country', how='left')
 
# Contrastive table (from clean title-gate outputs)
contrastive = pd.read_csv(
    'data/derived/step8_analysis_title_clean/contrastive_table_full_vs_edu_vs_title.csv'
)
 
# Stakeholder attrition
attrition = pd.read_csv(
    'data/derived/step8_analysis/stakeholder_chunks_vs_statements_edu.csv'
)
 
print("Data loaded.")
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4.3.1 — The inherited/direct split (proportional waffle chart)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 4.3.1 ...")
 
total       = 47512
edu_total   = 10557
title_total = 1795
inherited   = edu_total - title_total
non_edu     = total - edu_total
 
# Proportions
p_non    = non_edu / total        # 77.8%
p_inh    = inherited / total      # 17.8%
p_dir    = title_total / total    # 4.4%
 
# Build a 10x10 waffle: each cell = 1%
CELLS = 100
vals = {
    'Not education-relevant\n(77.8%)': round(p_non * CELLS),
    'Inherited education governance\n(17.8%)': round(p_inh * CELLS),
    'Direct education governance\n(3.8%)': CELLS - round(p_non*CELLS) - round(p_inh*CELLS),
}
colors_waffle = [C_GREY, C_BLUE, C_ORANGE]
labels_waffle = list(vals.keys())
counts_waffle = list(vals.values())
 
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.set_aspect('equal')
ax.axis('off')
 
cell = 0
COLS = 20
ROWS = 5
COLOR_MAP = []
for count, col in zip(counts_waffle, colors_waffle):
    COLOR_MAP.extend([col] * count)
 
for i, color in enumerate(COLOR_MAP):
    row = i // COLS
    col_pos = i % COLS
    rect = mpatches.FancyBboxPatch(
        (col_pos * 1.2, (ROWS - 1 - row) * 1.2),
        1.0, 1.0,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='white',
        linewidth=1.5
    )
    ax.add_patch(rect)
 
ax.set_xlim(-0.5, COLS * 1.2 + 0.5)
ax.set_ylim(-1.5, ROWS * 1.2 + 2)
 
# Legend
legend_items = [
    mpatches.Patch(facecolor=C_GREY,   edgecolor='white', label=f'Not education-relevant  (n={non_edu:,} chunks, 77.8%)'),
    mpatches.Patch(facecolor=C_BLUE,   edgecolor='white', label=f'Inherited education governance  (n={inherited:,} chunks, 17.8%)'),
    mpatches.Patch(facecolor=C_ORANGE, edgecolor='white', label=f'Direct education governance  (n={title_total:,} chunks, 3.8%)'),
]
ax.legend(
    handles=legend_items,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=1,
    frameon=False,
    fontsize=9.5
)
 
ax.set_title(
    'Distribution of the full corpus by education-governance type\n'
    '(each cell ≈ 1% of 47,512 paragraph-based chunks)',
    fontsize=11, fontweight='bold', pad=12
)
 
plt.tight_layout()
plt.savefig(OUT / 'fig4_3_1_inherited_direct_waffle.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig4_3_1_inherited_direct_waffle.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4.3.2 — Governance concentration: overall vs education (top 20)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 4.3.2 ...")
 
# Use top 20 by overall statements, compare with edu share
top20 = overall.sort_values('n_statements', ascending=False).head(20).copy()
top20['overall_pct'] = top20['n_statements'] / overall['n_statements'].sum() * 100
 
# Merge education data
edu_lookup = edu.set_index('country')['n_statements'].to_dict()
top20['edu_statements'] = top20['country'].map(edu_lookup).fillna(0)
top20['edu_pct'] = top20['edu_statements'] / edu['n_statements'].sum() * 100
 
fig, ax = plt.subplots(figsize=(10, 7))
 
y     = np.arange(len(top20))
h     = 0.35
bars1 = ax.barh(y + h/2, top20['overall_pct'], h, color=C_BLUE,   label='Overall AI governance', alpha=0.9)
bars2 = ax.barh(y - h/2, top20['edu_pct'],     h, color=C_ORANGE, label='Education-relevant governance', alpha=0.9)
 
ax.set_yticks(y)
ax.set_yticklabels(top20['country'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Share of global institutional statements (%)', fontsize=10)
ax.set_title(
    'Governance concentration: share of total institutional statements\n'
    'by jurisdiction (top 20 by overall output)',
    fontsize=11, fontweight='bold'
)
ax.legend(frameon=False, fontsize=9.5)
 
# Annotate the US shift
us_row = top20[top20['country'] == 'United States'].index[0]
us_pos = top20.index.get_loc(us_row)
ax.annotate(
    '28.6% (edu)\nvs 21.6% (overall)',
    xy=(top20.loc[us_row, 'edu_pct'], us_pos - h/2),
    xytext=(top20.loc[us_row, 'edu_pct'] + 1.5, us_pos - h/2 - 0.6),
    fontsize=8, color=C_ORANGE,
    arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=1)
)
 
plt.tight_layout()
plt.savefig(OUT / 'fig4_3_2_concentration_overall_vs_edu.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig4_3_2_concentration_overall_vs_edu.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4.3.3 — Governance thinness: statements per document
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 4.3.3 ...")
 
overall_w = overall.copy()
overall_w['stmts_per_doc_overall'] = overall_w['n_statements'] / overall_w['n_docs']
 
edu_w = edu.copy()
edu_w['stmts_per_doc_edu'] = edu_w['n_statements'] / edu_w['n_docs']
 
merged = overall_w[['country','stmts_per_doc_overall']].merge(
    edu_w[['country','stmts_per_doc_edu','n_statements']],
    on='country', how='inner'
)
 
# Label only notable countries
LABEL = {
    'United States', 'European Union', 'United Kingdom',
    'Türkiye', 'Australia', 'Malta', 'Norway', 'Colombia'
}
 
fig, ax = plt.subplots(figsize=(9, 7))
 
for _, row in merged.iterrows():
    size  = max(20, row['n_statements'] / 5)
    color = C_ORANGE if row['country'] in {'United States','European Union','United Kingdom'} else C_BLUE
    alpha = 0.85 if row['country'] in LABEL else 0.55
    ax.scatter(row['stmts_per_doc_overall'], row['stmts_per_doc_edu'],
               s=size, color=color, alpha=alpha, edgecolors='white', linewidths=0.5)
    if row['country'] in LABEL:
        ax.annotate(
            row['country'],
            (row['stmts_per_doc_overall'], row['stmts_per_doc_edu']),
            fontsize=7.5, ha='left', va='bottom',
            xytext=(4, 4), textcoords='offset points'
        )
 
# Reference lines for means
ax.axvline(overall_w['stmts_per_doc_overall'].mean(), color=C_BLUE, lw=1, ls='--', alpha=0.5,
           label=f'Overall mean ({overall_w["stmts_per_doc_overall"].mean():.0f} stmts/doc)')
ax.axhline(edu_w['stmts_per_doc_edu'].mean(), color=C_ORANGE, lw=1, ls='--', alpha=0.5,
           label=f'Education mean ({edu_w["stmts_per_doc_edu"].mean():.0f} stmts/doc)')
 
ax.set_xlabel('Overall statements per document', fontsize=10)
ax.set_ylabel('Education-relevant statements per document', fontsize=10)
ax.set_title(
    'Governance thinness: institutional density per document\n'
    '(bubble size proportional to education statement count)',
    fontsize=11, fontweight='bold'
)
ax.legend(frameon=False, fontsize=9)
 
patches = [
    mpatches.Patch(color=C_ORANGE, label='High-density jurisdictions (US, EU, UK)'),
    mpatches.Patch(color=C_BLUE,   label='All other jurisdictions'),
]
ax.legend(handles=patches + [
    plt.Line2D([0],[0], color=C_BLUE,   ls='--', label=f'Overall mean ({overall_w["stmts_per_doc_overall"].mean():.0f} stmts/doc)'),
    plt.Line2D([0],[0], color=C_ORANGE, ls='--', label=f'Education mean ({edu_w["stmts_per_doc_edu"].mean():.0f} stmts/doc)'),
], frameon=False, fontsize=8.5, loc='upper left')
 
plt.tight_layout()
plt.savefig(OUT / 'fig4_3_3_governance_thinness_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig4_3_3_governance_thinness_scatter.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4.3.4 — Statement type comparison across three corpora
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 4.3.4 ...")
 
corpora = ['Full corpus\n(874 docs)', 'Education-relevant\n(671 docs)', 'Education-directed\n(34 docs)']
strategy = [40.13, 42.66, 52.89]
norm     = [54.14, 51.28, 41.49]
rule     = [1.70,  2.23,  1.15]
other    = [4.03,  3.83,  4.47]
 
x   = np.arange(len(corpora))
w   = 0.18
 
fig, ax = plt.subplots(figsize=(9, 6))
 
b1 = ax.bar(x - 1.5*w, strategy, w, label='Strategy',    color=C_GREY,   alpha=0.9)
b2 = ax.bar(x - 0.5*w, norm,     w, label='Norm',        color=C_BLUE,   alpha=0.9)
b3 = ax.bar(x + 0.5*w, rule,     w, label='Rule',        color=C_DARK,   alpha=0.9)
b4 = ax.bar(x + 1.5*w, other,    w, label='Other/low confidence', color=C_LIGHT, edgecolor=C_GREY, alpha=0.9)
 
# Value labels on strategy and norm bars
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 1:
            ax.text(
                bar.get_x() + bar.get_width()/2, h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8
            )
 
ax.set_xticks(x)
ax.set_xticklabels(corpora, fontsize=10)
ax.set_ylabel('Share of institutional statements (%)', fontsize=10)
ax.set_title(
    'Statement type distribution across three analytical corpora\n'
    'Strategy share rises and rule share falls as corpus narrows toward education-directed documents',
    fontsize=11, fontweight='bold'
)
ax.legend(frameon=False, fontsize=9.5, loc='upper right')
ax.set_ylim(0, 65)
 
# Arrow showing the aspiration gradient
ax.annotate(
    '', xy=(2 - 1.5*w + w/2, 57), xytext=(0 - 1.5*w + w/2, 44),
    arrowprops=dict(arrowstyle='->', color='#888', lw=1.5, connectionstyle='arc3,rad=0.1')
)
ax.text(1, 60, 'Increasing aspiration →', fontsize=8.5, color='#666', ha='center')
 
plt.tight_layout()
plt.savefig(OUT / 'fig4_3_4_statement_types_three_corpora.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig4_3_4_statement_types_three_corpora.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4.3.5 — Education governance cluster world map (plotly)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 4.3.5 ...")
 
# Map country names to ISO-3 codes
ISO3 = {
    'United States': 'USA', 'European Union': None, 'Türkiye': 'TUR',
    'United Kingdom': 'GBR', 'Australia': 'AUS', 'Malta': 'MLT',
    'Spain': 'ESP', 'Belgium': 'BEL', 'India': 'IND', 'Japan': 'JPN',
    'Czech Republic': 'CZE', 'Norway': 'NOR', 'Mexico': 'MEX',
    'Austria': 'AUT', 'France': 'FRA', 'Thailand': 'THA',
    'Portugal': 'PRT', 'Ireland': 'IRL', 'Serbia': 'SRB',
    'Finland': 'FIN', 'Germany': 'DEU', 'Italy': 'ITA',
    'Netherlands': 'NLD', 'Poland': 'POL', 'Sweden': 'SWE',
    'Denmark': 'DNK', 'Hungary': 'HUN', 'Romania': 'ROU',
    'Bulgaria': 'BGR', 'Croatia': 'HRV', 'Slovakia': 'SVK',
    'Estonia': 'EST', 'Latvia': 'LVA', 'Lithuania': 'LTU',
    'Slovenia': 'SVN', 'Luxembourg': 'LUX', 'Cyprus': 'CYP',
    'Greece': 'GRC', 'China': 'CHN', 'Brazil': 'BRA',
    'Canada': 'CAN', 'South Korea': 'KOR', 'Russia': 'RUS',
    'Singapore': 'SGP', 'New Zealand': 'NZL', 'Israel': 'ISR',
    'Switzerland': 'CHE', 'Chile': 'CHL', 'Argentina': 'ARG',
    'Colombia': 'COL', 'Peru': 'PER', 'Uruguay': 'URY',
    'Saudi Arabia': 'SAU', 'UAE': 'ARE', 'United Arab Emirates': 'ARE',
    'Qatar': 'QAT', 'Egypt': 'EGY', 'Morocco': 'MAR',
    'South Africa': 'ZAF', 'Kenya': 'KEN', 'Nigeria': 'NGA',
    'Rwanda': 'RWA', 'Ghana': 'GHA', 'Tunisia': 'TUN',
    'Ukraine': 'UKR', 'Kazakhstan': 'KAZ', 'Uzbekistan': 'UZB',
    'Iceland': 'ISL', 'Mauritius': 'MUS', 'Malaysia': 'MYS',
    'Indonesia': 'IDN', 'Vietnam': 'VNM', 'Philippines': 'PHL',
    'Pakistan': 'PAK', 'Bangladesh': 'BGD', 'Sri Lanka': 'LKA',
    'Costa Rica': 'CRI', 'Panama': 'PAN', 'Ecuador': 'ECU',
    'Bolivia': 'BOL', 'Paraguay': 'PRY', 'Venezuela': 'VEN',
    'Dominican Republic': 'DOM', 'Cuba': 'CUB', 'Honduras': 'HND',
    'Guatemala': 'GTM', 'El Salvador': 'SLV', 'Nicaragua': 'NIC',
    'Korea': 'KOR', 'Zambia': 'ZMB', 'Uganda': 'UGA',
    'Armenia': 'ARM', 'Georgia': 'GEO', 'Moldova': 'MDA',
    'Albania': 'ALB', 'North Macedonia': 'MKD', 'Montenegro': 'MNE',
    'Bosnia and Herzegovina': 'BIH', 'Kosovo': None,
    'Liechtenstein': 'LIE', 'Monaco': 'MCO', 'San Marino': 'SMR',
    'Andorra': 'AND', 'Malta': 'MLT',
    'OECD': None, 'G7': None, 'G20': None,
    'World Economic Forum': None, 'United Nations': None,
    'UNESCO': None, 'Council of Europe': None, 'African Union': None,
    'Global Partnership on Artificial Intelligence': None,
    'Inter-American Development Bank': None,
}
 
CLUSTER_LABELS = {
    0: 'Cluster 0: Higher regulatory intensity',
    1: 'Cluster 1: Norm-dominant, zero rules',
    2: 'Cluster 2: Strategy-dominant',
    3: 'Cluster 3: Modal architecture (45 countries)',
    4: 'Cluster 4: High-density (US, EU)',
}
 
CLUSTER_COLORS = {
    0: '#E07B3F',
    1: '#4CAF50',
    2: '#9C27B0',
    3: '#2E75B6',
    4: '#1F3864',
}
 
plot_data = clusters_edu.copy()
plot_data['iso3'] = plot_data['country'].map(ISO3)
plot_data = plot_data.dropna(subset=['iso3'])
plot_data['cluster_label'] = plot_data['cluster_id'].map(CLUSTER_LABELS)
plot_data['color'] = plot_data['cluster_id'].map(CLUSTER_COLORS)
 
fig_map = go.Figure()
 
for cluster_id in sorted(plot_data['cluster_id'].unique()):
    sub = plot_data[plot_data['cluster_id'] == cluster_id]
    fig_map.add_trace(go.Choropleth(
        locations=sub['iso3'],
        z=[cluster_id] * len(sub),
        colorscale=[[0, CLUSTER_COLORS[cluster_id]], [1, CLUSTER_COLORS[cluster_id]]],
        showscale=False,
        name=CLUSTER_LABELS[cluster_id],
        text=sub['country'],
        hovertemplate='<b>%{text}</b><br>Cluster: ' + CLUSTER_LABELS[cluster_id] + '<extra></extra>',
        marker_line_color='white',
        marker_line_width=0.5,
        zmin=0, zmax=4,
    ))


fig_map.update_layout(
    title=dict(
        text='Education governance profiles: k=5 cluster membership across 74 jurisdictions',
        font=dict(size=13, family=FONT),
        x=0.5
    ),
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='#CCCCCC',
        showland=True,
        landcolor='#F5F5F5',
        showocean=True,
        oceancolor='#EAF2FF',
        projection_type='natural earth',
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom', y=-0.15,
        xanchor='center', x=0.5,
        font=dict(size=9, family=FONT),
        bgcolor='rgba(0,0,0,0)'
    ),
    margin=dict(l=0, r=0, t=50, b=80),
    width=1100, height=600,
    paper_bgcolor='white',
    plot_bgcolor='white',
)
 
pio.write_image(fig_map, str(OUT / 'fig4_3_5_edu_cluster_worldmap.png'), scale=3)
print("  Saved fig4_3_5_edu_cluster_worldmap.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("All figures saved to:", OUT)
print()
for f in sorted(OUT.iterdir()):
    size = f.stat().st_size / 1024
    print(f"  {f.name:55s}  {size:7.1f} KB")
 
