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

## Inference
In the docker container, please execute
```
$ python3 transfer.py -c data/real_to_watercolor/config.yaml --skip_cropping
```