# Analysis Module

This directory contains analysis scripts for the Traffic Sign Recognition project.

## Scripts

### `data_analysis.py`
Analyzes the GTSRB dataset:
- Dataset statistics
- Class distribution
- ROI (Region of Interest) statistics
- Image statistics
- Generates visualization: `data_analysis.png`

**Usage:**
```bash
python bonus/analysis/data_analysis.py
```

### `compare_all_models.py`
Compares training results from all models:
- Performance metrics comparison
- Generalization analysis
- Training curves
- Generates visualization: `model_comparison.png`
- Saves summary: `model_comparison_summary.json`

**Usage:**
```bash
python bonus/analysis/compare_all_models.py
```

**Prerequisites:** Run training for all models first:
```bash
python bonus/scripts/train_all_models.py
```

## Workflow

1. **Analyze dataset:**
   ```bash
   python bonus/analysis/data_analysis.py
   ```

2. **Train all models:**
   ```bash
   python bonus/scripts/train_all_models.py --epoch 5
   ```

3. **Compare all models:**
   ```bash
   python bonus/analysis/compare_all_models.py
   ```

## Output Files

- `data_analysis.png` - Dataset statistics visualization
- `model_comparison.png` - Model comparison visualization
- `model_comparison_summary.json` - Detailed comparison metrics

