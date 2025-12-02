# Spiking-Mix: A Modified Regularization Method Designed for Spiking Neural Network
#### This repo holds the codes for Spiking-Mix.

#### LIFNode
[NeuBridge: bridging quantized activations and spiking neurons for ANN-SNN conversion](https://github.com/Intelli-Chip-Lab/NeuBridge)
<br>We use the model in this paper and **quantify values of activated neurons**. 
<br>This model still uses the **LIF model** to accumulate the membrane voltage.
<br>This can more effectively **reduce the time-step required for accumulating**.

#### Architectures
For network architectures, we support **ResNet and Wide-ResNet**. 

#### Block Structure
Shi, Xinyu,Hao, Zecheng,Yu, Zhaofei.SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks[C].//IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).2024:5610-5619.
<br>Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pp. 448–456, 2015.

## Center Point Distance Distribution
$$
\begin{align}
    \sqrt{\frac{(x^2 + y^2)}{(H^2 + W^2)}} & = \sqrt{\frac{(x_2 - x_1)^2 - (y_2 - y_1)^2}{H^2 + W^2}} \\ 
              & = \sqrt{\frac{((\gamma * x_2 + \delta) - x_1)^2 - ((\gamma * y_2 + \delta) - y_1)^2}{H^2 + W^2}} \\ 
    \gamma    & = \epsilon + \frac{1 - \theta * \lambda^2 + \theta * \delta}{\epsilon + \theta * (1 - \lambda)^2 + \theta * e^-\delta}\\
\end{align}
$$

## Credits
The code for neuron models is based on the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repo. 
<br>The code for some utils is from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repo.

# Spiking-Mix
