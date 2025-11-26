# QR Code Scaling Experiment

This project investigates how the size of the **minimum logical element** in a QR code—commonly referred to as the **module**—affects decoding accuracy.  
A *module* is a single cell within a matrix barcode symbol, representing one bit of encoded information.

## Objectives

- Analyze the relationship between module size (image scale) and QR code decoding accuracy.  
- Measure decoding time as a function of image scale.  
- Generate visual plots demonstrating these dependencies.  
- Identify practical thresholds and parameters relevant for various barcode scanning systems.

## Practical Value

The results of this study are valuable for engineers, system integrators, and users of scanning systems, providing insights into optimal QR code sizing and robustness under real-world conditions.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/qr-open-cv-experiments.git
cd qr-open-cv-experiments
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

If you do not use a `requirements.txt`, install packages manually:

```bash
pip install opencv-python matplotlib numpy
```
