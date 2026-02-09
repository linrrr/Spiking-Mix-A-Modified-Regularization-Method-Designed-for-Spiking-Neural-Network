# Spiking-Mix: A Modified Regularization Method Designed for Spiking Neural Network
#### This repo holds the codes for Spiking-Mix.

#### LIFNode
[NeuBridge: bridging quantized activations and spiking neurons for ANN-SNN conversion](https://github.com/Intelli-Chip-Lab/NeuBridge)
<br>We use the model in this paper and **quantify values of activated neurons**. 
<br>**Quantization Coding(Before the Threshold) - Rate Coding + Delay Coding(During the Threshold) - Time Coding(After the Threshold)**
<br><br>[The design of multi-input LIF neuron circuit based on novel memristor](https://mc.spacejournal.cn/article/doi/10.19304/J.ISSN1000-7180.2023.0891)
<br>**Multi-threshold characteristics**
<br><br>[Encoding](https://universal-lin.notion.site/Quantization-Coding-2cd196b34f21800093bafe1e56e68712)
<br><br>This model still uses the **LIF model** to accumulate the membrane voltage.
<br>This can more effectively **reduce the time-step required for accumulating**.

#### Architectures
[SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks](https://github.com/xyshi2000/SpikingResformer)
<br>张新岩,祝勇俊,吴宏杰,等. 基于并行Transformer和CNN的图像压缩感知重构网络[J]. 科技导报,2025, 43(2): 108-116;doi: 10.3981/j.issn.1000-7857.2023.12.01823
<br>Chen,Yinpeng,et al. "Dynamic relu." European conference on computer vision. Cham: Springer International Publishing, 2020.
<br>刘娟,谢梦瑶,袁佳俊,等.基于机器视觉和HRNet网络的人体背部穴位识别方法[J].机电工程技术,2026,55(01):102-108.

## Credits
The code for modules is based on the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repo. 
<br>The code for some utils is from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repo.

# Spiking-Mix
