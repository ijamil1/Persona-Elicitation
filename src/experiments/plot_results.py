import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_test_accuracies(icm_acc, golden_acc, chat_acc, pretrained_acc, country,
                         icm_std=0, golden_std=0, chat_std=0, pretrained_std=0, prefix=""):
    """
    Plot comparison of test accuracies across methods with error bars.
    """
    save_path = f"{prefix}_figure_1_persona_{country}.png"
    accuracies = [
        pretrained_acc * 100,
        chat_acc * 100,
        icm_acc * 100,
        golden_acc * 100,
    ]
    std_devs = [
        pretrained_std * 100,
        chat_std * 100,
        icm_std * 100,
        golden_std * 100,
    ]
    labels = [
        "Zero-shot (Pretrained)",
        "Zero-shot (Chat)",
        "ICM (Unsupervised)",
        "Golden Supervision",
    ]

    bar_colors = [
        "#9658ca",  # darker purple for zero-shot pretrained
        "#B366CC",  # purple for zero-shot chat
        "#58b6c0",  # teal for ICM
        "#FFD700",  # gold for golden supervision
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        x,
        accuracies,
        yerr=std_devs,
        capsize=5,
        color=bar_colors,
        tick_label=labels,
        edgecolor="k",
        zorder=2,
        error_kw={'elinewidth': 2, 'capthick': 2}
    )

    # Add hatching to zero-shot chat bar
    bars[1].set_hatch('...')

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(f"Test Accuracy Comparison - {country}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

    # Add value labels on bars (above error bars)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        err = std_devs[i]
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + err),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.grid(axis='y', zorder=1, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_accuracy_vs_num_examples(results, country, prefix=""):
    """
    Plot test accuracy as a function of number of in-context examples with error bars.

    Args:
        results: Dict with num_examples, gold_acc, gold_acc_std, random_acc, random_acc_std, etc.
        country: Country name for title
        prefix: Prefix for the output filename
    """
    save_path = f"{prefix}_figure_2_accuracy_vs_examples_{country}.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    num_examples = results['num_examples']
    gold_acc = [acc * 100 for acc in results['gold_acc']]
    gold_std = [std * 100 for std in results.get('gold_acc_std', [0] * len(num_examples))]
    random_acc = [acc * 100 for acc in results['random_acc']]
    random_std = [std * 100 for std in results.get('random_acc_std', [0] * len(num_examples))]

    # Plot gold labels with error bars
    ax.errorbar(num_examples, gold_acc, yerr=gold_std,
                fmt='o-', color='#FFD700', linewidth=2, markersize=8,
                label='Gold Labels', capsize=4, capthick=2, elinewidth=2)

    # Only plot ICM if icm_acc exists in results and is non-empty
    if 'icm_acc' in results and results['icm_acc']:
        icm_acc = [acc * 100 for acc in results['icm_acc']]
        icm_std = [std * 100 for std in results.get('icm_acc_std', [0] * len(num_examples))]
        ax.errorbar(num_examples, icm_acc, yerr=icm_std,
                    fmt='s-', color='#58b6c0', linewidth=2, markersize=8,
                    label='ICM Labels', capsize=4, capthick=2, elinewidth=2)

    # Plot random labels with error bars
    ax.errorbar(num_examples, random_acc, yerr=random_std,
                fmt='^-', color='#9658ca', linewidth=2, markersize=8,
                label='Random Labels (accuracy-matched)', capsize=4, capthick=2, elinewidth=2)

    ax.set_xlabel("Number of In-Context Examples", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(f"Test Accuracy vs Number of Examples - {country}", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits with some padding
    all_accs = results['gold_acc'] + results['random_acc']
    all_stds = results.get('gold_acc_std', []) + results.get('random_acc_std', [])
    if 'icm_acc' in results and results['icm_acc']:
        all_accs += results['icm_acc']
        all_stds += results.get('icm_acc_std', [])

    min_acc = min(all_accs) * 100
    max_acc = max(all_accs) * 100
    max_std = max(all_stds) * 100 if all_stds else 0
    padding = max((max_acc - min_acc) * 0.1, max_std) if max_acc > min_acc else 5
    ax.set_ylim(max(0, min_acc - padding), min(100, max_acc + padding + max_std))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from benchmark results JSON")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to JSON file containing benchmark results")
    args = parser.parse_args()

    # Extract filename prefix from input path (without .json extension)
    filename_prefix = os.path.basename(args.input).replace('.json', '')

    # Load JSON data
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Process each result object
    for result in data:
        country = result['country']

        # Extract mean accuracies and std devs for plot_test_accuracies
        golden_acc = result['golden']
        golden_std = result.get('golden_std_dev', 0)
        chat_acc = result['chat']
        chat_std = result.get('chat_std_dev', 0)
        pretrained_acc = result['pretrained']
        pretrained_std = result.get('pretrained_std_dev', 0)
        icm_acc = result.get('icm', 0)
        icm_std = result.get('icm_std_dev', 0)

        # Generate test accuracies bar plot with error bars
        plot_test_accuracies(
            icm_acc, golden_acc, chat_acc, pretrained_acc, country,
            icm_std=icm_std, golden_std=golden_std, chat_std=chat_std, pretrained_std=pretrained_std,
            prefix=filename_prefix
        )

        # Generate accuracy vs num_examples plot if comparison data exists
        comparison = result.get('comparison')
        if comparison is not None:
            plot_accuracy_vs_num_examples(comparison, country, prefix=filename_prefix)
        else:
            print(f"Skipping accuracy vs num_examples plot for {country} (no comparison data)")


if __name__ == "__main__":
    main()
