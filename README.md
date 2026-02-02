# QR Code Recognition Experiment

This project investigates how the size of the **minimum logical element** in a QR code—commonly referred to as the **module**—affects decoding accuracy.  
A *module* is a single cell within a matrix barcode symbol, representing one bit of encoded information.

## Practical Value

The results of this study are valuable for engineers, system integrators, and users of scanning systems, providing insights into optimal QR code sizing and robustness under real-world conditions.

## Installation

1) Clone the repository


```bash
git clone https://github.com/StudentSK0/qr-cv-research
cd qr-cv-research
````

2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
````

3) Install required dependencies

```bash
pip install -r requirements.txt
```



4) Also install system dependency for ZBar

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y libzbar0

# macOS
brew install zbar
```

5) Run web UI

```bash
python3 -m src.qr_core.web_app
```

## Expected Dataset Structure

```
datasets/
├── "Name_of_dataset"/                       
   ├── images/
   │   └── QR_CODE/               # QR code images (.jpg/.png)
   │       ├── 001.jpg
   │       ├── 002.jpg
   │       └── ...
   │
   └── markup/
       └── QR_CODE/               # Image annotations (.json)
           ├── 001.jpg.json
           ├── 002.jpg.json
           └── ...
```


