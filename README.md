# QR Code Recognition Experiment

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


## Project Structure

```
qr_open_cv_proj/
├── .venv/                             # Python virtual environment
│
├── datasets/                          # Datasets with QR images and annotations
│
├── outputs/                           # Generated processing results and graphics
│   ├── open_cv_json_and_graphics/
│   └── zxing_json_and_graphics/
│
├── src/                               # Python source files
│   ├── open_cv_programs/
│   │   ├── qr_plot_results_open_cv.py
│   │   └── qr_process_data_open_cv.py
│   │
│   └── zxing_programs/
│       ├── qr_plot_results_zxing.py
│       └── qr_process_data_zxing.py
│
├── .gitignore                         # Git ignore rules
└── README.md                          # Project documentation

```

## datasets Directory Structure
```
datasets/
├── Dubska/                       
│   ├── images/
│   │   └── QR_CODE/               # QR code images (.jpg/.png)
│   │       ├── 001.jpg
│   │       ├── 002.jpg
│   │       └── ...
│   │
│   └── markup/
│       └── QR_CODE/               # Image annotations (.json)
│           ├── 001.jpg.json
│           ├── 002.jpg.json
│           └── ...
│
└── SE-Barcode/                    
    ├── images/
    │   └── QR_CODE/               # QR code images
    │
    └── markup/
        └── QR_CODE/               # Image annotations

```

