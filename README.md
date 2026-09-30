# AlphaForge Trading Model

Machine learning model developed for the AlphaForge Trading Competition.

## Approach

The model uses historical OHLCV market data and creates the following features:

- Return
- Price relative to a 3-period rolling average
- 3-period volatility
- Volume change

A Linear Regression model is used to generate continuous trading signals for each asset.

## Model Output

The model produces one continuous signal per asset in the range:

-1.0 to 1.0

## Files

- `model_random.py` — final competition model
- `README.md` — project documentation

## Requirements

```bash
pip install numpy pandas scikit-learn

