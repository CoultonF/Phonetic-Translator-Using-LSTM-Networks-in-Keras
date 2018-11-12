# Senior-Project

To train the project use Anaconda 4.3.30 on a CUDA 9.2 enabled machine using Tensorflow with support of Python 3.6.7

[How to setup TensorFlow with GPU support](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc)

To activate Tensorflow in the anaconda environment.
```
source activate tf_gpu
```

To run the project.
```
python cgan.py
```

To deactivate Tensorflow in the anaconda environment.
```
source deactivate tf_gpu
```

[*Modeled off of a previous implementation of a CDCGAN using MNIST data.*](https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN)
