import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_test_accuracies(icm_acc, golden_acc, chat_acc, pretrained_acc, country):
    """
    Plot comparison of test accuracies across methods.
    """
    save_path = f"figure_1_persona_{country}.png"
    accuracies = [
        pretrained_acc * 100,
        chat_acc * 100,
        icm_acc * 100,
        golden_acc * 100,
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
        color=bar_colors,
        tick_label=labels,
        edgecolor="k",
        zorder=2
    )

    # Add hatching to zero-shot chat bar
    bars[1].set_hatch('...')

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(f"Test Accuracy Comparison - {country}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.grid(axis='y', zorder=1, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_accuracy_vs_num_examples(results, country):
    """
    Plot test accuracy as a function of number of in-context examples.

    Args:
        results: Dict with num_examples, gold_acc, icm_acc, random_acc lists
        country: Country name for title
    """
    save_path = f"figure_2_accuracy_vs_examples_{country}.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    num_examples = results['num_examples']

    ax.plot(num_examples, [acc * 100 for acc in results['gold_acc']],
            'o-', color='#FFD700', linewidth=2, markersize=8, label='Gold Labels')

    # Only plot ICM if icm_acc exists in results
    if 'icm_acc' in results and results['icm_acc']:
        ax.plot(num_examples, [acc * 100 for acc in results['icm_acc']],
                's-', color='#58b6c0', linewidth=2, markersize=8, label='ICM Labels')

    ax.plot(num_examples, [acc * 100 for acc in results['random_acc']],
            '^-', color='#9658ca', linewidth=2, markersize=8, label='Random Labels (accuracy-matched)')

    ax.set_xlabel("Number of In-Context Examples", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(f"Test Accuracy vs Number of Examples - {country}", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits with some padding
    all_accs = results['gold_acc'] + results['random_acc']
    if 'icm_acc' in results and results['icm_acc']:
        all_accs += results['icm_acc']
    min_acc = min(all_accs) * 100
    max_acc = max(all_accs) * 100
    padding = (max_acc - min_acc) * 0.1 if max_acc > min_acc else 5
    ax.set_ylim(max(0, min_acc - padding), min(100, max_acc + padding))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from benchmark results JSON")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to JSON file containing benchmark results")
    args = parser.parse_args()

    # Load JSON data
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Process each result object
    for result in data:
        country = result['country']

        # Extract accuracies for plot_test_accuracies
        golden_acc = result['golden']
        chat_acc = result['chat']
        pretrained_acc = result['pretrained']
        icm_acc = 0  # Placeholder for now

        # Generate test accuracies bar plot
        plot_test_accuracies(icm_acc, golden_acc, chat_acc, pretrained_acc, country)

        # Generate accuracy vs num_examples plot if comparison data exists
        comparison = result.get('comparison')
        if comparison is not None:
            plot_accuracy_vs_num_examples(comparison, country)
        else:
            print(f"Skipping accuracy vs num_examples plot for {country} (no comparison data)")


if __name__ == "__main__":
    main()
