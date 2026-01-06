#!/usr/bin/env python3
"""
Create publication-quality comparison figure for report.
Shows old hard-stop vs new adaptive path management.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Path Management Strategy Comparison', fontsize=16, fontweight='bold', y=0.98)

# Simulated entropy sequence (with clear high-entropy positions)
positions = np.arange(0, 35)
np.random.seed(42)
entropy_base = 0.03 + 0.02 * np.random.rand(35)
# Add peaks at specific positions
entropy_base[5] = 0.08
entropy_base[8] = 0.09
entropy_base[12] = 0.07
entropy_base[20] = 0.12  # High entropy late in sequence
entropy_base[25] = 0.10  # High entropy late in sequence
entropy_base[30] = 0.11  # High entropy late in sequence

threshold = 0.055
max_paths = 8
branch_factor = 3

# ============================================================================
# Panel 1: OLD STRATEGY (Hard Stop)
# ============================================================================

ax1.set_title('(a) Traditional Hard-Stop Strategy', fontweight='bold', pad=10)

# Plot entropy
ax1.plot(positions, entropy_base, 'b-', linewidth=2, label='Normalized Entropy', zorder=2)
ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
            label=f'Threshold (τ={threshold})', zorder=1)

# Simulate old strategy branching
active_paths_old = [1]
branch_positions_old = []
blocked_positions_old = []

current_paths = 1
for i in range(1, len(positions)):
    if entropy_base[i] >= threshold:
        if current_paths < max_paths:
            # Can branch
            branch_positions_old.append(i)
            current_paths = min(current_paths * branch_factor, max_paths)
        else:
            # BLOCKED!
            blocked_positions_old.append(i)
    active_paths_old.append(current_paths)

# Mark branch points (green stars)
if branch_positions_old:
    branch_entropies = [entropy_base[i] for i in branch_positions_old]
    ax1.scatter(branch_positions_old, branch_entropies,
               marker='*', s=300, c='green', edgecolors='darkgreen',
               linewidths=1.5, label='Branched', zorder=5)

# Mark blocked points (red X)
if blocked_positions_old:
    blocked_entropies = [entropy_base[i] for i in blocked_positions_old]
    ax1.scatter(blocked_positions_old, blocked_entropies,
               marker='X', s=200, c='red', edgecolors='darkred',
               linewidths=2, label='Blocked (max_paths reached)', zorder=5)

# Add shaded region for "missed exploration"
if blocked_positions_old:
    first_blocked = min(blocked_positions_old)
    ax1.axvspan(first_blocked, positions[-1], alpha=0.1, color='red',
                label='Missed exploration zone')

ax1.set_ylabel('Normalized Entropy', fontsize=12)
ax1.set_ylim(-0.01, 0.16)
ax1.legend(loc='upper right', framealpha=0.95, fontsize=9)
ax1.grid(True, alpha=0.3, linestyle=':')

# Add text annotation
ax1.text(0.02, 0.95, f'Branch points: {len(branch_positions_old)}',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax1.text(0.02, 0.85, f'Blocked points: {len(blocked_positions_old)}',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# ============================================================================
# Panel 2: NEW STRATEGY (Adaptive Budgeting)
# ============================================================================

ax2.set_title('(b) Adaptive Budgeting Strategy', fontweight='bold', pad=10)

# Plot entropy
ax2.plot(positions, entropy_base, 'b-', linewidth=2, label='Normalized Entropy', zorder=2)
ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
            label=f'Threshold (τ={threshold})', zorder=1)

# Simulate new strategy branching
active_paths_new = [1]
branch_positions_new = []
branch_factors_used = []

current_paths = 1
for i in range(1, len(positions)):
    if entropy_base[i] >= threshold:
        # ALWAYS can branch (no hard stop)
        branch_positions_new.append(i)

        # Determine branch factor
        remaining_budget = max_paths - current_paths
        if remaining_budget >= branch_factor:
            bf = branch_factor
        elif remaining_budget > 0:
            bf = remaining_budget
        else:
            bf = 2  # Minimal branching

        branch_factors_used.append(bf)
        current_paths = min(current_paths + bf - 1, max_paths)

    active_paths_new.append(current_paths)

# Mark branch points with different colors based on branch factor
if branch_positions_new:
    for pos, bf in zip(branch_positions_new, branch_factors_used):
        if bf == branch_factor:
            color = 'green'
            marker = '*'
            size = 300
            label_text = 'Full branching'
        elif bf > 1:
            color = 'orange'
            marker = '*'
            size = 250
            label_text = 'Reduced branching'
        else:
            color = 'yellow'
            marker = '*'
            size = 200
            label_text = 'Minimal branching'

        ax2.scatter([pos], [entropy_base[pos]],
                   marker=marker, s=size, c=color, edgecolors='black',
                   linewidths=1.5, zorder=5)

# Create legend with proper line and marker styles
legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, label='Normalized Entropy'),
    Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', label=f'Threshold (τ={threshold})'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
           markeredgecolor='black', markersize=12, linewidth=0,
           label=f'Full branching (×{branch_factor})'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
           markeredgecolor='black', markersize=11, linewidth=0,
           label='Reduced branching (×1-2)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow',
           markeredgecolor='black', markersize=10, linewidth=0,
           label='Minimal + pruning (×2)')
]

ax2.set_xlabel('Token Position', fontsize=12)
ax2.set_ylabel('Normalized Entropy', fontsize=12)
ax2.set_ylim(-0.01, 0.16)
ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=9)
ax2.grid(True, alpha=0.3, linestyle=':')

# Add text annotation
ax2.text(0.02, 0.95, f'Branch points: {len(branch_positions_new)}',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax2.text(0.02, 0.85, f'Coverage: {(branch_positions_new[-1] - branch_positions_new[0]) if branch_positions_new else 0} positions',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()

# Save figure
output_path = 'path_management_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to: {output_path}")

# Also save as PDF for publication
output_pdf = 'path_management_comparison.pdf'
plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
print(f"✓ PDF version saved to: {output_pdf}")

# Create summary table
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"\n{'Metric':<30} {'Old Strategy':<20} {'New Strategy':<20}")
print("-" * 70)
print(f"{'Branch points':<30} {len(branch_positions_old):<20} {len(branch_positions_new):<20}")
print(f"{'Blocked points':<30} {len(blocked_positions_old):<20} {'0':<20}")
if branch_positions_old:
    old_span = branch_positions_old[-1] - branch_positions_old[0]
else:
    old_span = 0
if branch_positions_new:
    new_span = branch_positions_new[-1] - branch_positions_new[0]
else:
    new_span = 0
print(f"{'Branching span (positions)':<30} {old_span:<20} {new_span:<20}")
print(f"{'Exploration coverage':<30} {f'{old_span/35*100:.1f}%':<20} {f'{new_span/35*100:.1f}%':<20}")
print("-" * 70)
improvement = ((len(branch_positions_new) - len(branch_positions_old)) / len(branch_positions_old) * 100) if len(branch_positions_old) > 0 else 0
print(f"\n→ Branching improvement: +{improvement:.0f}%")
print(f"→ Coverage improvement: +{(new_span - old_span) / old_span * 100:.0f}%" if old_span > 0 else "")
print("=" * 70)

plt.show()
