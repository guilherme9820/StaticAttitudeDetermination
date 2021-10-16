[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0.0-%23FF6F00.svg?style=plastic&logo=TensorFlow&logoColor=white)](https://github.com/tensorflow/tensorflow/releases/tag/v2.0.0)
[![TensorFlow-Addons](https://img.shields.io/badge/TensorFlow_addons-0.6.0-%23FF6F00.svg?style=plastic&logo=TensorFlow&logoColor=white)](https://github.com/tensorflow/addons/releases/tag/v0.6.0)
[![Python](https://img.shields.io/badge/python-3.6-3670A0?style=plastic&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/release/python-360/)

# Static Attitude Determination Using Convolutional Neural Networks

Created by Guilherme Henrique dos Santos at [SpaceLab](https://spacelab.ufsc.br/en/home/) at Federal University of Santa Catarina (UFSC).

### Introduction

The need to estimate the orientation between frames of reference is crucial in spacecraft navigation. Robust algorithms for this type of problem have been built by following algebraic approaches, but data-driven solutions are becoming more appealing due to their stochastic nature. Hence, an approach based on convolutional neural networks in order to deal with measurement uncertainty in static attitude determination problems is proposed in this paper. PointNet models were trained with different datasets containing different numbers of observation vectors that were used to build attitude profile matrices, which were the inputs of the system. The uncertainty of measurements in the test scenarios was taken into consideration when choosing the best model. The proposed model, which used convolutional neural networks, proved to be less sensitive to higher noise than traditional algorithms, such as singular value decomposition (SVD), the q-method, the quaternion estimator (QUEST), and the second estimator of the optimal quaternion (ESOQ2). [MDPI](https://www.mdpi.com/1424-8220/21/19/6419)

| ![PoseCNN](https://https://github.com/guilherme9820/StaticAttitudeDetermination/architecture.png?raw=true) |
|:--:| 
| The proposed architecture (*B-Swish*) |

### License

This project is released under the MIT License (refer to the LICENSE file for details).

### Model training
1. To start the model training just run 'main.py' file passing the training parameters
    ```Shell
    python main.py --config attitude_params.yml
    ```

### Model testing
1. There are four testing scripts in total: 'angular_difference', 'dropout_evolution', 'test_scenarios', and 'uncertainty_test'. Each one evaluates a specific information. To run them, just run the 'main.py' file passing the testing parameters along with the test name.
    ```Shell
    python main.py --config tests_params.yml --test_name [TEST_NAME]
    ```

### Citation

If you find B-Swish useful in your research, please consider citing:

    @article{dos2021static,
        title={Static Attitude Determination Using Convolutional Neural Networks},
        author={dos Santos, Guilherme Henrique and Seman, Laio Oriel and Bezerra, Eduardo Augusto and Leithardt, Valderi Reis Quietinho and Mendes, Andr{\'e} Sales and Stefenon, St{\'e}fano Frizzo},
        journal={Sensors},
        volume={21},
        number={19},
        pages={6419},
        year={2021},
        publisher={Multidisciplinary Digital Publishing Institute}
    }