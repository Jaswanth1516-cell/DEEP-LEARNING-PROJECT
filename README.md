# DEEP-LEARNING-PROJECT

This project implements a deep learning image classification pipeline using PyTorch.

## What is included

- `train_model.py`: trains a CNN on the FashionMNIST dataset
- `requirements.txt`: needed Python dependencies
- `outputs/`: created by the training script and contains model weights and visualizations

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training:

```bash
python train_model.py --epochs 5 --batch-size 64
```

3. View output files in `outputs/`:

- `fashion_mnist_cnn.pth` - saved model weights
- `training_history.png` - loss and accuracy curves
- `sample_predictions.png` - example test images with predicted labels

## Notes

- The script automatically downloads the FashionMNIST dataset.
- It uses GPU if available, otherwise falls back to CPU.
