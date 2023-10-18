def make_CNN(droprate):
    """
    Implement a CNN Network using the following specification.
    The learned model is based on 2D convolutional neural network with dropout and ReLU as the activation function. The specific specification is `Conv2d(1, 8, 3)-Conv2d(8, 8, 3)- Conv2d(8, 16, 3)-Conv2d(16, 16, 3)-Conv2d(16, 32, 3)-Conv2d(32, 32, 3)-Conv2d(32, 64, 3)-Conv2d(64, 64, 3)-Conv2d(64, 128, 1)- Conv2d(128, 128, 1)-Conv2d(128, 1, 1)`  Conv2d(Cin, Cout, K) is a two-dimensional convolutional layer with Cin input channels, Cout output channels, a kernel size of K × K, a stride of (1,1), and no padding. All but the last convolutional layer are followed by a rectified linear unit (ReLU) and then a dropout layer with a dropout rate of 0.2. The size of the input spectrogram patch is set to that of the receptive field of the network (i.e., 17 × 17) so that the CNN output size is 1 × 1.
    """
    #todo 