# Spiking-Mix: A Modified Regularization Method Designed for Spiking Neural Network
#### This repo holds the codes for Spiking-Mix.

#### LIFNode
[NeuBridge: bridging quantized activations and spiking neurons for ANN-SNN conversion](https://github.com/Intelli-Chip-Lab/NeuBridge)
<br>We referenced the model in this paper and **activated neurons by quantifying the input values**. 
<br>This model still uses the **LIF model** to accumulate the membrane voltage.
<br>This method can more effectively **reduce the time-step required for accumulating**.
#### Architectures
For network architectures, we support **ResNet and Wide-ResNet**. 

## Credits
The code for data preprocessing and neuron models is based on the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repo. 
<br>The code for some utils is from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repo.

# Spiking-Mix
