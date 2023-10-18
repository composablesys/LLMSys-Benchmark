from generated import *  

import torch

def make_CNN_golden(droprate):
    """
    Implement a CNN  Network using the following specification.
    The learned model is based on 2D convolutional neural network with dropout and ReLU as the activation function. The specific specification is `Conv2d(1, 8, 3)-Conv2d(8, 8, 3)- Conv2d(8, 16, 3)-Conv2d(16, 16, 3)-Conv2d(16, 32, 3)-Conv2d(32, 32, 3)-Conv2d(32, 64, 3)-Conv2d(64, 64, 3)-Conv2d(64, 128, 1)- Conv2d(128, 128, 1)-Conv2d(128, 1, 1)`  Conv2d(Cin, Cout, K) is a two-dimensional convolutional layer with Cin input chatorch.nnels, Cout output chatorch.nnels, a kernel size of K × K, a stride of (1,1), and no padding. All but the last convolutional layer are followed by a rectified linear unit (ReLU) and then a dropout layer with a dropout rate of droprate. The size of the input spectrogram patch is set to that of the receptive field of the network (i.e., 17 × 17) so that the Ctorch.nn output size is 1 × 1.
    """

    layers = [
        torch.nn.Conv2d(1, 8, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(8, 8, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(8, 16, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(16, 16, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(16, 32, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(32, 32, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(32, 64, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(64, 64, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(64, 128, 1, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(128, 128, 1, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout(droprate),
        
        torch.nn.Conv2d(128, 1, 1, stride=1, padding="same")
    ]
    return torch.nn.Sequential(*layers) 


def test():
    for i in range(ITERATIONS):
        size = (16, 1, 513, 196)
        x = torch.rand(*size)
        golden = make_CNN_golden(0.3)
        golden_y =golden.forward(x)
        yhat = make_CNN(0.3).forward(x)
        torch.testing.assert_close(golden_y,yhat)
    return True

if __name__ == "__main__":
    print(test())