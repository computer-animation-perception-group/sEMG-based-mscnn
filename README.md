# A Multi-stream Convolutional Neural Network for sEMG-based Gesture Recognition in Musclecomputer interface

This repo contains the code for the experiments in the paper: Wentao Wei, Yongkang Wong, Yu Du, Mohan Kankanhalli, Weidong Geng. " [A Multi-stream Convolutional Neural Network for sEMG-based Gesture Recognition in Muscle-Computer Interface](https://www.sciencedirect.com/science/article/abs/pii/S0167865517304439)"

## Requirements
- A CUDA compatible GPU
- Ubuntu >= 14.04 or any other Linux/Unix that can run Docker
- [Docker](http://docker.io/)
- [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)

## Usage
- **Pull or build docker image**
    ``` 
    docker pull zjucapg/semg:latest
    ```
    or
    ``` 
    docker build -t zjucapg/semg:latest -d docker/Dockerfile .
    ```
- **Dataset**
    
    Three databases including Ninapro DB1, CapgMyo and CSL-HDEMG can be used for training and test.

    ```
    mkdir .cache
    # put NinaPro DB1 in .cache/ninapro-db1
    # put CapgMyo DB-a in .cache/dba or DB-b in .cache/dbb or DB-c in .cache/dbc
    # put CSL-HDEMG in .cache/csl
    ```
    The NinaPro DB1 needs to be segmented by gesture labels and stored in Matlab format as follows.`.cache/ninapro-db1/data/sss/ggg/sss_ggg_ttt.mat` contains a field `data` reprensents the trial `ttt` of gesture `ggg` of subject `sss`. And numbers start from zero. Gesture 0 is the rest gesture.

    For instance, `.cache/ninapro-db1/data/000/001/000_001_000.mat` is the 0th trial of 1st gesture of the 0th subject. 
    
    You can download the original dataset from <https://www.idiap.ch/project/ninapro/database> or download the prepared dataset from our site <http://zju-capg.org/myo/data/ninapro-db1.zip>. CapgMyo and CSL-HDEMG datasets can be acquired on <http://zju-capg.org/myo/data> and <http://www.csl.uni-bremen.de/cms/forschung/bewegungserkennung>, respectively.

- **Quick Start**
    ```
    # Get into the capg/semg:mscnn container
    nvidia-docker run -ti -v your_projectdir:/code zjucapg/semg /bin/bash
    # Train
    ./exp_ninapro    # Ninapro DB1
    or ./exp_dbc     # CapgMyo DB-c
    or ./exp_csl     # CSL HDEMG
    # Test
    python scripts/test_ninapro_multistream.py      # Ninapro DB1
    python scripts/test_dbc_multistream.py          # CapgMyo DB-c
    python scripts/test_csl_multistream.py          # CSL HDEMG
    ```


- **Trained model**
    We also provide trained model for straight using, which you can just extract the zip file and put it into `.cache`.
    - The model files are stored on Google Drive, which contain three categories as follows:
        - [Ninapro DB1](https://drive.google.com/open?id=1oc2smnwt5JuKqpOX9BSIqwf496x_3neE)
        - [CapgMyo DB-c](https://drive.google.com/open?id=1ehHyfhxoZnIwCXh1qkrWXrKhoj5ABaAS)
        - [CSL-HDEMG](https://drive.google.com/open?id=1GgAz1QwjPWtvfwU2O-mCOcp4-5odXgTb)

## License
Licensed under an GPL v3.0 license.

## Bibtex
```
@article{WEI20190301,
title = "A multi-stream convolutional neural network for sEMG-based gesture recognition in muscle-computer interface",
journal = "Pattern Recognition Letters",
volume = "119",
pages = "131 - 138",
year = "2019",
note = "Deep Learning for Pattern Recognition",
issn = "0167-8655",
doi = "https://doi.org/10.1016/j.patrec.2017.12.005",
url = "http://www.sciencedirect.com/science/article/pii/S0167865517304439",
author = "Wentao Wei and Yongkang Wong and Yu Du and Yu Hu and Mohan Kankanhalli and Weidong Geng"
}
```