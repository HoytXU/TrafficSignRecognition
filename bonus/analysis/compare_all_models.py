"""
Compare All Models Analysis

Analyzes and compares training results from all models.
"""

import os
import sys
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add bonus directory to path
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)

from core.config import AVAILABLE_MODELS, CHECKPOINT_DIR, LOGS_DIR


def parse_training_log(log_file):
    """Parse training log to extract metrics."""
    if not os.path.exists(log_file):
        return None
    
    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_f1': [],
        'best_test_acc': 0.0,
        'best_test_f1': 0.0,
        'final_train_acc': 0.0,
        'final_train_loss': 0.0
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract epoch-by-epoch metrics
        epoch_pattern = r'Epoch (\d+)/(\d+) - Train Loss: ([\d.]+) - Train Accuracy: ([\d.]+)'
        test_pattern = r'Test Accuracy: ([\d.]+) - Test F1-score \(macro\): ([\d.]+)'
        
        epochs = re.findall(epoch_pattern, content)
        tests = re.findall(test_pattern, content)
        
        for i, (epoch, total_epochs, loss, acc) in enumerate(epochs):
            if i < len(tests):
                test_acc, test_f1 = tests[i]
                metrics['epochs'].append(int(epoch))
                metrics['train_loss'].append(float(loss))
                metrics['train_acc'].append(float(acc))
                metrics['test_acc'].append(float(test_acc))
                metrics['test_f1'].append(float(test_f1))
                
                if float(test_acc) > metrics['best_test_acc']:
                    metrics['best_test_acc'] = float(test_acc)
                    metrics['best_test_f1'] = float(test_f1)
        
        if metrics['train_acc']:
            metrics['final_train_acc'] = metrics['train_acc'][-1]
            metrics['final_train_loss'] = metrics['train_loss'][-1]
            
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None
    
    return metrics if metrics['epochs'] else None


def compare_all_models():
    """Compare all trained models."""
    
    print("="*70)
    print("Model Comparison Analysis")
    print("="*70)
    
    # Parse all training logs
    all_metrics = {}
    
    for model_name in AVAILABLE_MODELS:
        log_file = os.path.join(LOGS_DIR, f"{model_name}_training.log")
        metrics = parse_training_log(log_file)
        
        if metrics:
            all_metrics[model_name] = metrics
            print(f"✓ Loaded metrics for {model_name}")
        else:
            print(f"✗ No metrics found for {model_name}")
    
    if not all_metrics:
        print("\nNo training results found. Please train models first:")
        print("  python bonus/scripts/train_all_models.py")
        return
    
    print(f"\nFound results for {len(all_metrics)} models")
    print("="*70)
    
    # Create comparison table
    print("\n1. Model Performance Comparison")
    print("-" * 70)
    print(f"{'Model':<15} {'Best Test Acc':<15} {'Best F1':<12} {'Final Train Acc':<18} {'Final Loss':<12}")
    print("-" * 70)
    
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['best_test_acc'], reverse=True)
    
    for model_name, metrics in sorted_models:
        print(f"{model_name:<15} {metrics['best_test_acc']:<15.4f} {metrics['best_test_f1']:<12.4f} "
              f"{metrics['final_train_acc']:<18.4f} {metrics['final_train_loss']:<12.4f}")
    
    # Statistics
    print("\n2. Performance Statistics")
    print("-" * 70)
    best_accs = [m['best_test_acc'] for m in all_metrics.values()]
    best_f1s = [m['best_test_f1'] for m in all_metrics.values()]
    
    print(f"Best Test Accuracy:")
    print(f"  Max: {max(best_accs):.4f} ({sorted_models[0][0]})")
    print(f"  Min: {min(best_accs):.4f}")
    print(f"  Mean: {np.mean(best_accs):.4f}")
    print(f"  Std: {np.std(best_accs):.4f}")
    
    print(f"\nBest F1-score:")
    print(f"  Max: {max(best_f1s):.4f}")
    print(f"  Min: {min(best_f1s):.4f}")
    print(f"  Mean: {np.mean(best_f1s):.4f}")
    print(f"  Std: {np.std(best_f1s):.4f}")
    
    # Generalization analysis
    print("\n3. Generalization Analysis")
    print("-" * 70)
    print(f"{'Model':<15} {'Train-Test Gap':<18} {'Status':<15}")
    print("-" * 70)
    
    for model_name, metrics in sorted_models:
        gap = metrics['final_train_acc'] - metrics['best_test_acc']
        status = "Good" if gap < 0.1 else "Overfitting" if gap > 0.15 else "Moderate"
        print(f"{model_name:<15} {gap:<18.4f} {status:<15}")
    
    # Create visualizations
    print("\n4. Generating comparison visualizations...")
    create_comparison_plots(all_metrics, sorted_models)
    
    # Save summary
    summary = {
        'models': {name: {
            'best_test_acc': m['best_test_acc'],
            'best_test_f1': m['best_test_f1'],
            'final_train_acc': m['final_train_acc'],
            'final_train_loss': m['final_train_loss'],
            'train_test_gap': m['final_train_acc'] - m['best_test_acc']
        } for name, m in all_metrics.items()},
        'statistics': {
            'best_acc_max': float(max(best_accs)),
            'best_acc_min': float(min(best_accs)),
            'best_acc_mean': float(np.mean(best_accs)),
            'best_acc_std': float(np.std(best_accs)),
            'best_f1_max': float(max(best_f1s)),
            'best_f1_min': float(min(best_f1s)),
            'best_f1_mean': float(np.mean(best_f1s)),
            'best_f1_std': float(np.std(best_f1s))
        }
    }
    
    summary_file = os.path.join(bonus_dir, "analysis", "model_comparison_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to: {summary_file}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    
    print(f"\n📊 Top 3 Models:")
    for i, (model_name, metrics) in enumerate(sorted_models[:3], 1):
        print(f"  {i}. {model_name}: {metrics['best_test_acc']:.4f} accuracy, "
              f"F1={metrics['best_test_f1']:.4f}")


def create_comparison_plots(all_metrics, sorted_models):
    """Create comparison visualization plots."""
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(16, 12))
    
    model_names = [m[0] for m in sorted_models]
    best_accs = [m[1]['best_test_acc'] for m in sorted_models]
    best_f1s = [m[1]['best_test_f1'] for m in sorted_models]
    train_accs = [m[1]['final_train_acc'] for m in sorted_models]
    gaps = [m[1]['final_train_acc'] - m[1]['best_test_acc'] for m in sorted_models]
    
    # 1. Best Test Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.barh(model_names, best_accs, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Test Accuracy')
    ax1.set_title('Best Test Accuracy by Model')
    ax1.set_xlim([min(best_accs) * 0.95, max(best_accs) * 1.02])
    ax1.grid(True, alpha=0.3, axis='x')
    for i, (bar, acc) in enumerate(zip(bars, best_accs)):
        ax1.text(acc, i, f' {acc:.3f}', va='center', fontweight='bold')
    
    # 2. Best F1-score Comparison
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.barh(model_names, best_f1s, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('F1-score (macro)')
    ax2.set_title('Best F1-score by Model')
    ax2.set_xlim([min(best_f1s) * 0.95, max(best_f1s) * 1.02])
    ax2.grid(True, alpha=0.3, axis='x')
    for i, (bar, f1) in enumerate(zip(bars, best_f1s)):
        ax2.text(f1, i, f' {f1:.3f}', va='center', fontweight='bold')
    
    # 3. Train vs Test Accuracy
    ax3 = plt.subplot(2, 3, 3)
    x_pos = np.arange(len(model_names))
    width = 0.35
    ax3.barh(x_pos - width/2, train_accs, width, label='Train', alpha=0.7, color='orange')
    ax3.barh(x_pos + width/2, best_accs, width, label='Test', alpha=0.7, color='steelblue')
    ax3.set_yticks(x_pos)
    ax3.set_yticklabels(model_names)
    ax3.set_xlabel('Accuracy')
    ax3.set_title('Train vs Test Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Generalization Gap
    ax4 = plt.subplot(2, 3, 4)
    colors = ['green' if g < 0.1 else 'orange' if g < 0.15 else 'red' for g in gaps]
    bars = ax4.barh(model_names, gaps, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Train-Test Gap')
    ax4.set_title('Generalization Gap (Lower is Better)')
    ax4.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Overfitting threshold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.legend()
    
    # 5. Accuracy vs F1-score scatter
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(best_accs, best_f1s, s=100, alpha=0.6, c=range(len(model_names)), cmap='viridis')
    for i, name in enumerate(model_names):
        ax5.annotate(name, (best_accs[i], best_f1s[i]), fontsize=8, alpha=0.7)
    ax5.set_xlabel('Test Accuracy')
    ax5.set_ylabel('F1-score')
    ax5.set_title('Accuracy vs F1-score')
    ax5.grid(True, alpha=0.3)
    
    # 6. Training curves (if available)
    ax6 = plt.subplot(2, 3, 6)
    for model_name, metrics in sorted_models[:5]:  # Top 5 models
        if metrics['epochs']:
            ax6.plot(metrics['epochs'], metrics['test_acc'], 'o-', label=model_name, alpha=0.7, linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Test Accuracy')
    ax6.set_title('Training Curves (Top 5 Models)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(bonus_dir, "analysis", "model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    compare_all_models()

