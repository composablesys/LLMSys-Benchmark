from generated import *  


import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def si_sdr(true, est):

    para = (torch.sum(true * est, 1, keepdim = True) / torch.sum(true ** 2, 1, keepdim = True)) * true
    num = torch.sum(para ** 2, 1)
    dnm = torch.sum((est - para) ** 2, 1)

    return 10 * torch.log10(num / dnm)


@torch.no_grad()
def calc_si_sdri(true, est, mix):

    return si_sdr(true, est) - si_sdr(true, mix)



# Test cases
def test_si_sdr_equality():
    for _ in range(5):
        # Generate random input data for testing
        true = torch.rand(10)  # Replace '10' with the appropriate dimension
        est = torch.rand(10)   # Replace '10' with the appropriate dimension
        mix = torch.rand(10)   # Replace '10' with the appropriate dimension

        # Calculate the output of 'calc_si_sdri'
        si_sdri_output = calc_si_sdri(true, est, mix)

        # Calculate the output of the unknown function 'generated'
        generated_output = calc_si_sdri_generated(true, est, mix)

        # Compare the two outputs for equality
        if not torch.allclose(si_sdri_output, generated_output):
            return False

    return True




if __name__ == "__main__":
    # Run the test and print the result
    result = test_si_sdr_equality()
    print(result)