# Deeper-Networks-for-Image-Classification

Classification on MNIST and CIFAR-10 with Tensorflow.

## Abstract
This github repository addresses the task of image classification using deeper convolutional neural networks (CNNs), including VGG, ResNet, and GoogleNet. The primary datasets employed for this study are MNIST and CIFAR-10 to provide a comprehensive assessment of the chosen networks. I proposed CIFAR-10 optimized models using Upsampling module and constant learning rate
(CIFAR-VGG16 and CIFAR-ResNet-50) improve upon image classification compared to VGG16 and ResNet-50 with ImageNet trained weights (75.33% to 93.00% and 64.66% to 94.666%accuracy, respectively). I further improved classification of GoogleNet
on CIFAR-10 using MiniGoogleNet architecture that leveraged batch normalization and downsampled module(from 81.15% to 90.04%), helping further understand about applying three deeper
networks architectures on different datasets

* VGG-16
* ResNet-50
* GoogleNet

## Results

### MNIST
Top1 error rate on the MNIST benchmarks are reported. You may get different results when training your models with different random seed.
<img width="1108" alt="Screenshot 2023-09-18 at 19 07 28" src="https://github.com/minhanh151/Deeper-Networks-for-Image-Classification/assets/55950352/884a6547-68d8-47ad-b036-9d4f3f54d6f6">

Table 1 shows the accuracy-related to each model on the MNIST dataset. Pre-trained VGG-16 and ResNet-50 on ImageNet with only trainable top layers achieved
97.91% and 96.3% testing accuracy after 40 epochs with short
training time. The alternative models useing trainable
convolutional layers from the VGG-16 and ResNet-50 produced accuracy rates that were surpassing LeNet (99.36%
and 99.4% respectively). However, the ResNet-50 architecture’s
training time (1592s) is quite lengthy. This is understandable
given that ResNet-50 has a deeper architecture than VGG-16 with more parameters.

Unlike VGGNet and ResNet, GoogLeNet could be trained
on MNIST from scratch and yielded a high results of 99.26%.
GoogLeNet has a relatively smaller model capacity compared
to VGGNet and ResNet. This reduced capacity makes it easier
to train from scratch on smaller datasets like MNIST, as it has
a lower risk of overfitting. The complexity of VGGNet and
ResNet models may lead to challenges in training on smaller
datasets, requiring pre-training on larger datasets or extensive
data augmentation techniques. 

### CIFAR
Top1 error rate on the CIFAR-10 benchmarks are reported. You may get different results when training your models with different random seed.

<img width="1133" alt="Screenshot 2023-09-18 at 19 04 38" src="https://github.com/minhanh151/Deeper-Networks-for-Image-Classification/assets/55950352/7c351dd6-13cd-42a7-b97a-0cc2cd1b27cf">

Table 2 shows accuracy-related to each model with different
settings in architecture on the CIFAR-10 dataset. Transfer learning model of VGG-16
and ResNet-50 with untrainable convolutional layers yielded
poor results of 75.33% and 64.66% respectively. Freezing
the convolutional layers prevents them from being fine-tuned
on the CIFAR-10 dataset. VGG-16 and ResNet-50 are deep
and complex models with a large number of parameters.
With frozen convolutional layers, the models might lack the
flexibility to adapt and learn new representations specifically
suited for the CIFAR-10 dataset.

With trainable convolutional layers and upsampling module,
ResNet-50 produces the highest accuracy of 94.66% after only
3 epochs. Training VGG-16 and ResNet-50 from pre-trained
weights can get a good initialization and enhance their ability
to capture the dataset’s unique features and patterns on CIFAR10 effectively.

We can also see MiniGoogleNet performs better than GoogleNet on the CIFAR-10 dataset (90.04% and 81.15%),
but all models start to overfit slightly after 40 epochs. CIFAR10 is a relatively small dataset, and the model’s capacity might
be too high, leading to overfitting.

## Further Evaluation
* Upsampling2D technique
on pretrained VGG-16 and ResNet-50 models significantly
improves the accuracy rate on the CIFAR-10 dataset. CIFAR10 images have a relatively low resolution (32x32 pixels),
which can result in a loss of spatial information during
downsampling operations like pooling or strided convolutions.
Upsampling2D helps recover some of that lost information by
increasing the spatial dimensions of feature maps, allowing
the network to capture more detailed patterns and fine-grained
information.
* The implementation of a very low constant learning rate is
also one of the important things during transfer learning of
VGG-16 and ResNet-50 on CIFAR-10. This is likely because
the model was trained using Imagent, and the steps to apply
gradient descent should not be too large because we might
enter a zone that is not the actual minimum value.





