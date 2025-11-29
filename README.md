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
   \sqrt{\frac{(x^2 + y^2)}{(H^2 + W^2)}}
   & = \sqrt{\frac{((x_2 - x_1)^2 - (y_2 - y_1)^2)}{(H^2 + W^2)}} \\
   & = \sqrt{\frac{((\gamma * x_2 + \delta) - x_1)^2 - ((\gamma * y_2 + \delta) - y_1)^2)}{(H^2 + W^2)}} \\

   \gamma = \epsilon + \frac{{(1 - \theta * lam^2 + \theta * region)} {(\epsilon + \theta * (1 - lam)^2 + \theta * e^(-region^2))}}
\end{align}
$$

# Spiking-Mix
