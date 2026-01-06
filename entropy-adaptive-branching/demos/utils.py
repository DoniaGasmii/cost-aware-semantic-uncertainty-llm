"""
Visualization utilities for EAB demos.

Functions for creating entropy plots, sample trees, and resource comparisons.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict


def plot_entropy_vs_tokens(samples, threshold, entropy_data=None, save_path=None):
    """
    Plot entropy over token positions with threshold line.

    Shows where entropy spikes occur and which positions triggered branching.

    Args:
        samples: List of sample dictionaries
        threshold: Entropy threshold used for branching
        entropy_data: Optional entropy history from eab.get_entropy_history()
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use entropy_data if provided, otherwise try to extract from samples
    if entropy_data and 'positions' in entropy_data and 'entropies' in entropy_data:
        # Use the entropy history from EAB
        positions = entropy_data['positions']
        entropies = entropy_data['entropies']
        branched = entropy_data.get('branched', [])

        # Plot entropy line
        ax.plot(positions, entropies, color='darkblue', linewidth=2, label='Entropy')

        # Mark branch points with stars
        branch_positions = [pos for pos, did_branch in zip(positions, branched) if did_branch]
        if branch_positions:
            branch_entropies = [ent for ent, did_branch in zip(entropies, branched) if did_branch]
            ax.scatter(branch_positions, branch_entropies, color='red', s=200,
                      zorder=5, marker='*', label='Branch Points', edgecolors='darkred', linewidth=1.5)

        # Draw threshold line
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

        # Styling
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Normalized Entropy', fontsize=12)
        ax.set_title('Entropy vs Token Position (Real-Time Tracking)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return

    # Fallback: Try to collect entropy data from samples (legacy)
    all_entropies = []
    max_len = 0

    for sample in samples:
        entropy_hist = sample.get('entropy_history', [])
        if entropy_hist:
            all_entropies.append(entropy_hist)
            max_len = max(max_len, len(entropy_hist))

    if not all_entropies:
        # Fallback: If no entropy history, create visualization from branch points
        branch_points = set()
        for sample in samples:
            branch_points.update(sample.get('branch_points', []))

        if branch_points:
            # Show branch points as vertical lines
            for pos in branch_points:
                ax.axvline(x=pos, color='red', alpha=0.3, linestyle='--', linewidth=1)

            ax.axhline(y=threshold, color='orange', linestyle='-', linewidth=2, label=f'Threshold ({threshold})')
            ax.set_xlabel('Token Position', fontsize=12)
            ax.set_ylabel('Entropy (inferred)', fontsize=12)
            ax.set_title('Branching Positions (entropy history not available)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            return

        # If no data at all, show warning
        ax.text(0.5, 0.5, 'No entropy data available\n(EAB may not have tracked entropy history)',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return

    # Plot each sample's entropy history
    for i, entropy_hist in enumerate(all_entropies):
        positions = list(range(len(entropy_hist)))
        ax.plot(positions, entropy_hist, alpha=0.3, color='blue', linewidth=1)

    # Plot average entropy
    # Pad shorter sequences with NaN
    padded_entropies = []
    for ent_hist in all_entropies:
        padded = ent_hist + [np.nan] * (max_len - len(ent_hist))
        padded_entropies.append(padded)

    avg_entropy = np.nanmean(padded_entropies, axis=0)
    positions = list(range(len(avg_entropy)))
    ax.plot(positions, avg_entropy, color='darkblue', linewidth=2, label='Average Entropy')

    # Mark branch points
    branch_points = set()
    for sample in samples:
        branch_points.update(sample.get('branch_points', []))

    for pos in branch_points:
        if pos < len(avg_entropy):
            ax.scatter([pos], [avg_entropy[pos]], color='red', s=100, zorder=5, marker='*', label='Branch' if pos == min(branch_points) else '')

    # Draw threshold line
    ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

    # Styling
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title('Entropy vs Token Position', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_sample_tree(samples, save_path=None):
    """
    Visualize the tree structure of samples showing parent-child relationships.

    Uses path_id and parent_id to construct the branching tree.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Build tree structure
    # samples have path_id and parent_id
    tree = defaultdict(list)  # parent_id -> list of children
    all_path_ids = set()
    all_parent_ids = set()

    for sample in samples:
        path_id = sample.get('path_id', 0)
        parent_id = sample.get('parent_id', None)

        all_path_ids.add(path_id)
        if parent_id is not None:
            all_parent_ids.add(parent_id)
            tree[parent_id].append(path_id)
        else:
            # Root node
            tree[-1].append(path_id)

    # Find root nodes: path_ids that are not in any sample (virtual roots)
    # or path_ids that have no parent
    if not tree[-1]:  # No explicit root nodes found
        # Find nodes that are parents but not children (these are virtual roots)
        root_candidates = all_parent_ids - all_path_ids
        if root_candidates:
            # These are virtual root nodes - add them as starting points
            tree[-1] = sorted(list(root_candidates))
        else:
            # All nodes have parents within the set
            # This shouldn't happen in a proper tree, but handle it
            # Use path_ids with minimum parent_id
            if all_parent_ids and all_path_ids:
                min_parent = min(all_parent_ids)
                # Add children of the minimum parent as roots
                tree[-1] = tree.get(min_parent, sorted(list(all_path_ids))[:1])

    if not tree or len(samples) == 0:
        # No samples to plot
        ax.text(0.5, 0.5, 'No samples generated',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return

    # Layout tree using simple recursive approach
    positions = {}
    x_counter = [0]  # Use list to make it mutable in nested function

    def layout_tree(node_id, depth=0):
        """Recursively layout nodes."""
        if node_id in positions:
            return

        # Position this node
        x = x_counter[0]
        x_counter[0] += 1
        y = -depth  # Negative so tree grows downward

        positions[node_id] = (x, y)

        # Layout children
        children = tree.get(node_id, [])
        for child in sorted(children):
            layout_tree(child, depth + 1)

    # Start from root
    for root in tree.get(-1, []):
        layout_tree(root, depth=0)

    # Draw edges
    for parent, children in tree.items():
        if parent == -1:
            continue
        if parent not in positions:
            continue

        parent_pos = positions[parent]
        for child in children:
            if child in positions:
                child_pos = positions[child]
                ax.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]],
                        'k-', alpha=0.4, linewidth=1)

    # Draw nodes
    for node_id, (x, y) in positions.items():
        # Find corresponding sample
        sample = next((s for s in samples if s.get('path_id') == node_id), None)

        if sample:
            # Real sample node
            # Color based on number of branches
            num_branches = len(sample.get('branch_points', []))

            if num_branches == 0:
                color = 'lightgreen'
            elif num_branches <= 3:
                color = 'yellow'
            else:
                color = 'orange'

            ax.scatter([x], [y], s=500, c=color, edgecolors='black', linewidth=2, zorder=3)
            ax.text(x, y, f'P{node_id}', ha='center', va='center', fontsize=10, fontweight='bold')

            # Add branch count below node
            if num_branches > 0:
                ax.text(x, y - 0.15, f'({num_branches}br)', ha='center', va='top', fontsize=8, style='italic')
        else:
            # Virtual root node (e.g., initial prompt state)
            color = 'lightblue'
            ax.scatter([x], [y], s=400, c=color, edgecolors='blue', linewidth=2, zorder=3, marker='s')
            ax.text(x, y, 'Root', ha='center', va='center', fontsize=9, fontweight='bold')

    # Legend
    blue_patch = mpatches.Patch(color='lightblue', label='Root (prompt)', edgecolor='blue', linewidth=2)
    green_patch = mpatches.Patch(color='lightgreen', label='No branches')
    yellow_patch = mpatches.Patch(color='yellow', label='1-3 branches')
    orange_patch = mpatches.Patch(color='orange', label='4+ branches')
    ax.legend(handles=[blue_patch, green_patch, yellow_patch, orange_patch], loc='upper right')

    ax.set_title('Sample Tree Structure\n(P = Path ID, br = branches)', fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    # Adjust limits to fit all nodes
    if positions:
        xs, ys = zip(*positions.values())
        margin = 1
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_resource_comparison(eab_metrics, naive_metrics, save_path=None):
    """
    Compare resource usage between EAB and Naive sampling.

    Shows speedup in tokens, time, and memory.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['total_tokens', 'wall_time', 'peak_memory_mb']
    labels = ['Token-Steps', 'Wall Time (s)', 'Memory (MB)']
    colors = ['#3498db', '#e74c3c']

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]

        eab_val = eab_metrics.get(metric, 0)
        naive_val = naive_metrics.get(metric, 0)

        # Bar chart
        bars = ax.bar(['EAB', 'Naive'], [eab_val, naive_val], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Calculate speedup
        if eab_val > 0 and naive_val > 0:
            speedup = naive_val / eab_val
            ax.text(0.5, 0.95, f'Speedup: {speedup:.2f}×',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Resource Comparison: EAB vs Naive', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def save_samples_to_file(eab_samples, file_path, prompt, naive_samples=None):
    """
    Save all generated samples to a human-readable text file.

    Args:
        eab_samples: Samples from EAB generation
        file_path: Path to save file
        prompt: The input prompt used
        naive_samples: Optional naive samples for comparison
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("  Generated Samples Comparison\n")
        f.write("="*70 + "\n\n")
        f.write(f"Prompt: {prompt}\n\n")

        # EAB Samples Section
        f.write("="*70 + "\n")
        f.write(f"  EAB SAMPLES ({len(eab_samples)} total)\n")
        f.write("="*70 + "\n\n")

        for i, sample in enumerate(eab_samples, 1):
            path_id = sample.get('path_id', '?')
            parent_id = sample.get('parent_id', '?')
            num_branches = len(sample.get('branch_points', []))
            branch_points = sample.get('branch_points', [])

            f.write(f"[EAB Sample {i}] Path {path_id} (parent: {parent_id})\n")
            f.write(f"  Branches: {num_branches} at positions {branch_points}\n")
            f.write(f"  Text: {sample.get('text', sample.get('generated_only', 'N/A'))}\n")
            f.write("-"*70 + "\n\n")

        # Naive Samples Section (if provided)
        if naive_samples:
            f.write("\n" + "="*70 + "\n")
            f.write(f"  NAIVE SAMPLES ({len(naive_samples)} total)\n")
            f.write("="*70 + "\n\n")

            for i, sample in enumerate(naive_samples, 1):
                # Prioritize 'generated_only' to avoid showing full chat template
                text = sample.get('generated_only', sample.get('text', 'N/A'))
                num_tokens = sample.get('num_tokens', 'N/A')

                f.write(f"[Naive Sample {i}] ({num_tokens} tokens)\n")
                f.write(f"  Text: {text}\n")
                f.write("-"*70 + "\n\n")

        f.write("="*70 + "\n")
        f.write("  End of Comparison\n")
        f.write("="*70 + "\n")
