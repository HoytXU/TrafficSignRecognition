#!/usr/bin/env python3
"""
Train All Models Script

Trains all available models sequentially and saves results for comparison.
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# Add bonus directory to path
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)

from core.config import AVAILABLE_MODELS, MODEL_CONFIGS, DEFAULT_CONFIG, LOGS_DIR


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GRAY = '\033[90m'


def train_all_models(epochs=5, batch_size=128, weight_decay=0.001):
    """
    Train all available models.
    
    Args:
        epochs: Number of epochs to train each model
        batch_size: Batch size for training
        weight_decay: Weight decay for optimizer
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    project_root = os.path.dirname(bonus_dir)
    train_script = os.path.join(bonus_dir, "training", "train.py")
    
    results = {}
    start_time = datetime.now()
    
    # Print header
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'═' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'Training All Models'.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'═' * 70}{Colors.ENDC}\n")
    print(f"{Colors.OKCYAN}ℹ Total models: {Colors.BOLD}{len(AVAILABLE_MODELS)}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}ℹ Epochs per model: {Colors.BOLD}{epochs}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}ℹ Batch size: {Colors.BOLD}{batch_size}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}ℹ Weight decay: {Colors.BOLD}{weight_decay}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}ℹ Start time: {Colors.BOLD}{start_time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    for i, model_name in enumerate(AVAILABLE_MODELS, 1):
        progress = i / len(AVAILABLE_MODELS) * 100
        bar_length = 30
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n{Colors.BOLD}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}[{i}/{len(AVAILABLE_MODELS)}] Training {model_name.upper()} [{bar}] {progress:.1f}%{Colors.ENDC}")
        print(f"{Colors.BOLD}{'─' * 70}{Colors.ENDC}\n")
        
        # Get model-specific learning rate
        lr = MODEL_CONFIGS[model_name].get('lr', DEFAULT_CONFIG['lr'])
        
        # Build command
        cmd = [
            sys.executable,
            train_script,
            "--model", model_name,
            "--epoch", str(epochs),
            "--lr", str(lr),
            "--batch_size", str(batch_size),
            "--weight_decay", str(weight_decay)
        ]
        
        # Run training
        log_file = os.path.join(LOGS_DIR, f"{model_name}_training.log")
        process = None
        try:
            # Use Popen to stream output in real-time while logging to file
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output to both terminal and file
                # Read line by line to preserve tqdm progress bars
                for line in iter(process.stdout.readline, ''):
                    # Write to terminal
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    # Write to log file
                    f.write(line)
                    f.flush()
                
                # Wait for process to complete
                returncode = process.wait()
            
            if returncode == 0:
                print(f"\n{Colors.OKGREEN}✓ {model_name.upper()} completed successfully{Colors.ENDC}")
                results[model_name] = {
                    'status': 'success',
                    'log_file': log_file
                }
            else:
                print(f"\n{Colors.FAIL}✗ {model_name.upper()} failed with return code {returncode} (check {log_file}){Colors.ENDC}")
                results[model_name] = {
                    'status': 'failed',
                    'returncode': returncode,
                    'log_file': log_file
                }
        except KeyboardInterrupt:
            if process:
                process.kill()
                process.wait()
            print(f"\n{Colors.WARNING}✗ {model_name.upper()} interrupted by user{Colors.ENDC}")
            results[model_name] = {
                'status': 'interrupted',
                'log_file': log_file
            }
            raise
        except Exception as e:
            print(f"\n{Colors.FAIL}✗ {model_name.upper()} error: {e}{Colors.ENDC}")
            results[model_name] = {
                'status': 'error',
                'error': str(e),
                'log_file': log_file
            }
        
        print()
    
    end_time = datetime.now()
    duration = end_time - start_time
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = sum(1 for r in results.values() if r['status'] != 'success')
    
    # Print summary
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'═' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'Training Summary'.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'═' * 70}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Results:{Colors.ENDC}")
    print(f"  {Colors.OKGREEN}✓ Successful: {Colors.BOLD}{successful}/{len(AVAILABLE_MODELS)}{Colors.ENDC}")
    if failed > 0:
        print(f"  {Colors.FAIL}✗ Failed: {Colors.BOLD}{failed}{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Timing:{Colors.ENDC}")
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        time_str = f"{int(minutes)}m {int(seconds)}s"
    else:
        time_str = f"{int(seconds)}s"
    print(f"  {Colors.GRAY}Total time: {time_str} ({duration.total_seconds():.2f}s){Colors.ENDC}")
    print(f"  {Colors.GRAY}Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"  {Colors.GRAY}End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    
    # Save results summary
    summary_file = os.path.join(LOGS_DIR, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'epochs': epochs,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'results': results
        }, f, indent=2)
    
    print(f"\n{Colors.BOLD}Logs:{Colors.ENDC}")
    print(f"  {Colors.OKCYAN}Summary saved to: {summary_file}{Colors.ENDC}")
    print(f"  {Colors.OKCYAN}Individual logs in: {LOGS_DIR}{Colors.ENDC}")
    
    if successful == len(AVAILABLE_MODELS):
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ All models trained successfully!{Colors.ENDC}")
    else:
        print(f"\n{Colors.WARNING}⚠ Some models failed. Check logs for details.{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Next step:{Colors.ENDC}")
    print(f"  {Colors.OKCYAN}python {os.path.join(bonus_dir, 'analysis', 'compare_all_models.py')}{Colors.ENDC}")
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'═' * 70}{Colors.ENDC}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all models')
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs per model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    
    args = parser.parse_args()
    
    train_all_models(
        epochs=args.epoch,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay
    )

