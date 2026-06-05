"""
High-Readability UML Class Diagram for SemantiQEngine Architecture
Run: python generate_uml_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Font & DPI ──────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# ── Colour palette ───────────────────────────────────────────────────────────
C = {
    'engine'      : ('#1565C0', '#E3F2FD'),   # header bg, body bg
    'ingestor'    : ('#2E7D32', '#E8F5E9'),
    'vectoriser'  : ('#6A1B9A', '#F3E5F5'),
    'rlagent'     : ('#BF360C', '#FBE9E7'),
    'sanitizer'   : ('#00695C', '#E0F2F1'),
    'arrow'       : '#37474F',
    'divider'     : '#90A4AE',
    'title_fg'    : 'white',
    'body_fg'     : '#212121',
    'attr_fg'     : '#37474F',
    'method_fg'   : '#1A237E',
    'bg'          : '#F8F9FA',
    'shadow'      : '#CFD8DC',
}

# ── Helper: draw one UML class box ───────────────────────────────────────────
def draw_class_box(ax, x, y, w, h,
                   class_name, stereotype,
                   attributes, methods,
                   header_color, body_color,
                   header_h=0.55):
    """
    Draws a UML class box.
    x, y  = bottom-left corner
    w, h  = width, total height
    """
    # Shadow
    ax.add_patch(FancyBboxPatch(
        (x + 0.04, y - 0.04), w, h,
        boxstyle="round,pad=0.05",
        facecolor=C['shadow'], edgecolor='none', zorder=1))

    # Body
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=body_color, edgecolor=header_color,
        linewidth=2.0, zorder=2))

    # Header band
    ax.add_patch(FancyBboxPatch(
        (x, y + h - header_h), w, header_h,
        boxstyle="round,pad=0.05",
        facecolor=header_color, edgecolor=header_color,
        linewidth=0, zorder=3))

    # Stereotype
    ax.text(x + w/2, y + h - header_h*0.28,
            f'«{stereotype}»',
            ha='center', va='center',
            fontsize=7.5, color='white',
            fontstyle='italic', zorder=4)

    # Class name
    ax.text(x + w/2, y + h - header_h*0.72,
            class_name,
            ha='center', va='center',
            fontsize=10.5, fontweight='bold',
            color='white', zorder=4)

    # Divider line (attributes / methods)
    n_attr = len(attributes)
    n_meth = len(methods)
    total_rows = n_attr + n_meth + 1          # +1 for the divider gap
    row_h = (h - header_h) / (total_rows + 1)

    # Attribute rows
    for i, attr in enumerate(attributes):
        row_y = y + h - header_h - (i + 1) * row_h - row_h * 0.1
        ax.text(x + 0.12, row_y,
                attr,
                ha='left', va='center',
                fontsize=8.2, color=C['attr_fg'],
                fontfamily='monospace', zorder=4)

    # Divider
    div_y = y + h - header_h - (n_attr + 0.8) * row_h
    ax.plot([x + 0.08, x + w - 0.08], [div_y, div_y],
            color=C['divider'], linewidth=0.8, zorder=4)

    # Method rows
    for j, meth in enumerate(methods):
        row_y = y + h - header_h - (n_attr + j + 1.6) * row_h
        ax.text(x + 0.12, row_y,
                meth,
                ha='left', va='center',
                fontsize=8.2, color=C['method_fg'],
                fontfamily='monospace', zorder=4)


# ── Helper: curved arrow with label ─────────────────────────────────────────
def draw_arrow(ax, x1, y1, x2, y2, label='',
               connectionstyle="arc3,rad=-0.25"):
    ax.annotate('',
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='-|>', color=C['arrow'],
            lw=1.8,
            connectionstyle=connectionstyle,
            mutation_scale=14),
        zorder=5)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.18, label,
                ha='center', va='bottom',
                fontsize=8, color=C['arrow'],
                fontstyle='italic',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=1.5),
                zorder=6)


# ════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor(C['bg'])
ax.set_facecolor(C['bg'])

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(8, 8.65,
        'SemantiQEngine  —  UML Class Diagram',
        ha='center', va='center',
        fontsize=14, fontweight='bold', color='#1A237E',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')])

# ════════════════════════════════════════════════════════════════════════════
# CLASS BOXES
# ════════════════════════════════════════════════════════════════════════════

# ── SemantiQEngine (centre-top) ──────────────────────────────────────────────
draw_class_box(
    ax, x=5.5, y=5.2, w=5.0, h=2.9,
    class_name='SemantiQEngine',
    stereotype='controller',
    attributes=[
        '- config : Dict',
        '- state : SystemState',
    ],
    methods=[
        '+ initialize()',
        '+ orchestrate()',
        '+ trigger_healing()',
    ],
    header_color=C['engine'][0],
    body_color=C['engine'][1],
    header_h=0.60,
)

# ── DataIngestor (bottom-left) ───────────────────────────────────────────────
draw_class_box(
    ax, x=0.4, y=0.6, w=3.5, h=3.8,
    class_name='DataIngestor',
    stereotype='component',
    attributes=[
        '- log_path : Path',
        '- raw_data : DataFrame',
    ],
    methods=[
        '+ parse_junit_xml()',
        '+ clean_history()',
    ],
    header_color=C['ingestor'][0],
    body_color=C['ingestor'][1],
)

# ── BertVectoriser (centre-left) ─────────────────────────────────────────────
draw_class_box(
    ax, x=4.25, y=0.6, w=3.5, h=3.8,
    class_name='BertVectoriser',
    stereotype='component',
    attributes=[
        '- model : BertModel',
        '- tokenizer : Tokenizer',
    ],
    methods=[
        '+ load_weights()',
        '+ generate_embedding()',
    ],
    header_color=C['vectoriser'][0],
    body_color=C['vectoriser'][1],
)

# ── RLAgent (centre-right) ───────────────────────────────────────────────────
draw_class_box(
    ax, x=8.25, y=0.6, w=3.5, h=3.8,
    class_name='RLAgent',
    stereotype='component',
    attributes=[
        '- policy_net : DQN',
        '- gamma : float',
    ],
    methods=[
        '+ predict_risk(state)',
        '+ update_q_values()',
    ],
    header_color=C['rlagent'][0],
    body_color=C['rlagent'][1],
)

# ── TopoSanitizer (bottom-right) ─────────────────────────────────────────────
draw_class_box(
    ax, x=12.1, y=0.6, w=3.5, h=3.8,
    class_name='TopoSanitizer',
    stereotype='component',
    attributes=[
        '- graph : DiGraph',
        '- visited : Set',
    ],
    methods=[
        '+ build_dag()',
        '+ topological_sort()',
    ],
    header_color=C['sanitizer'][0],
    body_color=C['sanitizer'][1],
)

# ════════════════════════════════════════════════════════════════════════════
# ARROWS  (engine bottom → each component top)
# ════════════════════════════════════════════════════════════════════════════

# Engine centre-bottom
ex, ey = 8.0, 5.2          # engine bottom-centre

# DataIngestor top-centre
draw_arrow(ax, ex, ey, 2.15, 4.4,
           label='1. ingests',
           connectionstyle='arc3,rad=0.30')

# BertVectoriser top-centre
draw_arrow(ax, ex, ey, 6.0, 4.4,
           label='2. vectorises',
           connectionstyle='arc3,rad=0.15')

# RLAgent top-centre
draw_arrow(ax, ex, ey, 10.0, 4.4,
           label='3. ranks',
           connectionstyle='arc3,rad=-0.15')

# TopoSanitizer top-centre
draw_arrow(ax, ex, ey, 13.85, 4.4,
           label='4. sanitises',
           connectionstyle='arc3,rad=-0.30')

# ════════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C['engine'][0],     'SemantiQEngine  (Controller)'),
    (C['ingestor'][0],   'DataIngestor'),
    (C['vectoriser'][0], 'BertVectoriser'),
    (C['rlagent'][0],    'RLAgent'),
    (C['sanitizer'][0],  'TopoSanitizer'),
]
for i, (color, label) in enumerate(legend_items):
    lx, ly = 0.45, 8.45 - i * 0.38
    ax.add_patch(mpatches.Rectangle(
        (lx, ly - 0.12), 0.28, 0.24,
        facecolor=color, edgecolor='none', zorder=5))
    ax.text(lx + 0.38, ly,
            label, va='center', fontsize=8.5,
            color=C['body_fg'], zorder=5)

# Legend border
ax.add_patch(FancyBboxPatch(
    (0.3, 6.5), 4.0, 2.2,
    boxstyle="round,pad=0.1",
    facecolor='white', edgecolor=C['divider'],
    linewidth=1.0, alpha=0.85, zorder=4))
ax.text(2.3, 8.58, 'Legend',
        ha='center', fontsize=9, fontweight='bold',
        color=C['body_fg'], zorder=5)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig('semantiq_uml_diagram.png', dpi=300, bbox_inches='tight',
            facecolor=C['bg'])
print("✓  Saved: semantiq_uml_diagram.png")
plt.close()
