"""Compare training results from all models."""

import os
import re
import json
import matplotlib.pyplot as plt
import importlib.util

# Load config
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("config", os.path.join(bonus_dir, "core", "config.py"))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


def parse_log(log_file):
    """Extract metrics from training log."""
    if not os.path.exists(log_file):
        return None
    
    ansi = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    epochs, test_accs, test_f1s, train_accs, train_losses = [], [], [], [], []
    current_epoch = None
    
    with open(log_file, 'r') as f:
        for line in f:
            clean = ansi.sub('', line)
            
            # Track epoch
            if match := re.search(r'Epoch (\d+)/(\d+)', clean):
                epoch_num = int(match.group(1))
                if any(x in clean for x in ['[', '%', '─', '═']) or 'Train Loss' in clean:
                    current_epoch = epoch_num
            
            # Extract metrics
            if match := re.search(r'Test Acc(uracy)?:\s*([\d.]+)', clean):
                if current_epoch:
                    epochs.append(current_epoch)
                    test_accs.append(float(match.group(2)))
            if match := re.search(r'Test F1(-score \(macro\))?:\s*([\d.]+)', clean):
                test_f1s.append(float(match.group(2)))
            if match := re.search(r'Train Acc(uracy)?:\s*([\d.]+)', clean):
                train_accs.append(float(match.group(2)))
            if match := re.search(r'Train Loss:\s*([\d.]+)', clean):
                train_losses.append(float(match.group(1)))
    
    if not test_accs:
        return None
    
    # Ensure epochs match test_accs length
    if len(epochs) != len(test_accs):
        epochs = list(range(1, len(test_accs) + 1))
    
    best_idx = test_accs.index(max(test_accs))
    return {
        'best_test_acc': max(test_accs),
        'best_test_f1': test_f1s[best_idx] if test_f1s else 0,
        'best_epoch': epochs[best_idx],
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'test_accs': test_accs,
        'epochs': epochs,
    }


def main():
    """Compare all models and generate visualization."""
    # Parse logs
    metrics = {m: parse_log(os.path.join(config.LOGS_DIR, f"{m}_training.log"))
               for m in config.AVAILABLE_MODELS}
    metrics = {k: v for k, v in metrics.items() if v}
    
    if not metrics:
        print("No training results found.")
        return
    
    # Print table
    print(f"\n{'Model':<15} {'Test Acc':<12} {'F1':<10} {'Epoch':<8} {'Train Acc':<12} {'Loss':<10}")
    print("-" * 75)
    for model, m in sorted(metrics.items(), key=lambda x: x[1]['best_test_acc'], reverse=True):
        print(f"{model:<15} {m['best_test_acc']:<12.4f} {m['best_test_f1']:<10.4f} "
              f"Epoch {m['best_epoch']:<5} {m['final_train_acc']:<12.4f} {m['final_train_loss']:<10.4f}")
    
    # Create plots
    sorted_models = sorted(metrics.items(), key=lambda x: x[1]['best_test_acc'], reverse=True)
    names = [m[0] for m in sorted_models]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Accuracy & F1 bars
    for ax, data, color, title in zip(axes[:2], 
                                       ([m[1]['best_test_acc'] for m in sorted_models],
                                        [m[1]['best_test_f1'] for m in sorted_models]),
                                       ['steelblue', 'green'],
                                       ['Test Accuracy', 'F1-score']):
        ax.barh(names, data, color=color, alpha=0.7)
        ax.set_xlabel(title)
        ax.set_title(f'Model Comparison: {title}')
        ax.set_xlim([min(data) * 0.95, max(data) * 1.02])
        for i, val in enumerate(data):
            ax.text(val, i, f' {val:.3f}', va='center')
    
    # Training curves
    for model, m in sorted_models:
        if m['epochs'] and m['test_accs']:
            axes[2].plot(m['epochs'], m['test_accs'], 'o-', label=model, alpha=0.7, linewidth=2, markersize=4)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Test Accuracy')
    axes[2].set_title('Training Curves: Test Accuracy')
    axes[2].legend(fontsize=8, loc='best')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output = os.path.join(bonus_dir, "analysis", "model_comparison.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output}")
    
    # Save summary
    summary = {name: {k: v for k, v in m.items() if k not in ['test_accs', 'epochs']} 
               for name, m in metrics.items()}
    with open(os.path.join(bonus_dir, "analysis", "model_comparison_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {os.path.join(bonus_dir, 'analysis', 'model_comparison_summary.json')}")


if __name__ == "__main__":
    main()
