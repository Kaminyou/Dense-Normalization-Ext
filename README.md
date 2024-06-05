# Dense-Normalization
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
![version](https://img.shields.io/badge/version-0.1.0-red)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Kaminyou/Dense-Normalization/blob/main/LICENSE)
<!-- ![linting workflow](https://github.com/Kaminyou/Dense-Normalization/actions/workflows/main.yml/badge.svg) -->

## Environment preparation
1. Please check your GPU driver version and modify `Dockerifle` accordingly
2. Then, execute
    ```
    $ docker-compose up --build -d
    ```
3. Get into the docker container
    ```
    $ docker exec -it dn-env bash
    ```

## Config example
```yml
INFERENCE_SETTING:
  TEST_X: "./data_ori/real_to_watercolor/pair_03_0104_gt.jpg"
  TEST_DIR_X: "./data/real_to_watercolor/test_pair_03_0104_gt/"
  MODEL_VERSION: "20"
  SAVE_ORIGINAL_IMAGE: False
  NORMALIZATION: "dn"
  INTERPOLATE_MODE: "bicubic"
  PARALLELISM: True
```

## Inference
In the docker container, please execute
```
$ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 transfer.py -c data/real_to_watercolor/config.yaml --skip_cropping
```

## Visualization
In the docker container, please execute
```
$ ./run_jupyter.sh
```
Then open your `19555` port (please modify `docker-compose.yml` if needed) and click `Visualization.ipynb`.