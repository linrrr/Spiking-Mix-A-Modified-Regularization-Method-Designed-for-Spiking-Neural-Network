# Spiking-Mix: A Modified Regularization Method Designed for Spiking Neural Network
#### This repo holds the codes for Spiking-Mix.

#### LIFNode
[NeuBridge: bridging quantized activations and spiking neurons for ANN-SNN conversion](https://github.com/Intelli-Chip-Lab/NeuBridge)
<br>We use the model in this paper and **quantify values of activated neurons**. 
<br>This model still uses the **LIF model** to accumulate the membrane voltage.
<br>This can more effectively **reduce the time-step required for accumulating**.
#### Architectures
For network architectures, we support **ResNet and Wide-ResNet**. 

## Credits
The code for neuron models is based on the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repo. 
<br>The code for some utils is from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repo.

## Center Point Distance Distribution
$$
\begin{align}
    \sqrt{\frac{(x^2 + y^2)}{(H^2 + W^2)}} & = \sqrt{\frac{73^2-1}{12^2}} \\
              & = \sqrt{\frac{73^2}{12^2}\cdot\frac{73^2-1}{73^2}} \\ 
              & = \sqrt{\frac{73^2}{12^2}}\sqrt{\frac{73^2-1}{73^2}} \\
              & = \frac{73}{12}\sqrt{1-\frac{1}{73^2}} \\ 
              & \approx \frac{73}{12}\left(1-\frac{1}{2\cdot73^2}\right) \\
\end{align}
$$

# Spiking-Mix
