# QBMG: Quasi-Biogenic Molecule Generator with Deep Recurrent Network 
## Introduction
This repository contains scripts for generating biogenic-like moulecules. The user can (i) obatin a certain number of unbiased biogenic-like molecules and (ii) obtain a certain number of focused chemotype derivatives with transfer learning.We thank [the previous work by the Olivecrona team](https://github.com/MarcusOlivecrona/REINVENT).The code in this repository is inspired on REINVENT.

## Usage
- sample.py: script that is used to generate a certain number of molecules (Both unbiased or certain chemotype biogenic molecules) with trained model. 

  **Example:sampling 1000 molecules from unbiased model and focused chemotype model respectively**
``` Python
   %run sample.py ./data/biogenic.ckpt 1000
```
``` Python
   %run sample.py coumarin.ckpt 1000
```

- transfer_learning.py: script that is used to train focused chemotype biogenic moleucles and obtain focused chemotype derivatives. The user can provide own focused chemotype molecules to generate new derivates.

  **Example:training the focused chemotype molecules**
``` Python
   %run transfer_learning.py ./data/coumarin.smi 
```

## Requirments
This package requires:
- Python 3.6
- PyTorch 0.4.0
- RDKit
- tqdm
- jupyter notebook

## Contact
Welcome to contact us.
http://www.rcdd.org.cn/home/

