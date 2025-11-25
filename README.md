# EdgeLUT

## Usage
### Dependency
```
conda create -n edgelut python=3.8
conda activate edgelut
pip install -r requirements.txt
```

```
project-root/
│
├── 📂 checkpoints/              
│
├── 📂 dataset/                    
│
├── 📂 models/                  #model definition related
│
├── 📂 luts/                  
│
├── 📂 scripts/               
│
├── 📂 sr/               # for super resolution application
│
├── 📂 log/              
│
├── 📂 dn/                 # for dnoising application
│
├── 📂 dblur/                # for deblurring application
│
├── 📂 dmosaic/               # for deblocking application
│
│
├── transfer.py               # transfer from model to LUT
├── requirements.txt       
└── README.md
```
