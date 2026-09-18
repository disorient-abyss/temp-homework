# 归一化

- Normalization 是一类技术的统称，核心目标是将激活值拉回到均值为 0、方差为 1 的稳定分布，通常包含标准化和仿射变换两个阶段：
  
  - $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$
    - （其中 $\mu$ 和 $\sigma^2$ 的统计维度取决于具体的归一化方法，$\gamma$ 与 $\beta$ 为可学习的缩放与偏置参数。）

## 1 BatchNorm(BN，批量归一化)

- 机制：沿 Batch 维度计算（每个通道独立统计同一批次中所有样本的均值与方差），强依赖于 Batch size。

- 局限：
  
    1. 强依赖 Batch 大小，小 Batch 时统计量极不稳定；
  
    2. 变长序列处理困难：批次中短序列引入的 Padding 补零会严重污染统计量；
  
    3. 训练与推理机制割裂：训练期使用当前 Batch 的统计量，推理期依赖全局滑动平均统计量（running_mean / running_var）。

## 2  LayerNorm(LN，层归一化)

- 机制：沿 Feature / Hidden 维度计算，每个样本（或 Token）独立统计自身所有特征的均值与方差。

- 优势：
  
    1. 计算完全在单个样本/Token 内部闭环，训练与推理行为完全一致；
  
    2. 天然解耦 Batch 维度，不受 Batch size 限制；
  
    3. 原生适配变长序列，能够彻底规避 Padding 带来的统计污染。

## 3 架构演进：Post-LN vs Pre-LN

PyTorch 核心接口：

```
torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
```

```
Post-LN:  x_{l+1} = LayerNorm(x_l + SubLayer(x_l))
Pre-LN:   x_{l+1} = x_l + SubLayer(LayerNorm(x_l))
```

- **Post-LN（标准原始 Transformer）**：
  
  - 残差相加后立即进行 LayerNorm。
  
  - **痛点**：主残差通路上存在 LayerNorm 的非线性压缩，导致深层网络反向传播时梯度流衰减严重，极度依赖严格设计的 Warmup 学习率策略，难以直接训练百层以上深层网络。

- **Pre-LN（现代大模型基线）**：
  
  - 在进入多头注意力或 FFN 子层前先进行 LayerNorm，残差流本身保持无障碍直连。
  
  - **优势**：梯度能够通过恒等通路无衰减地直接流向浅层，训练稳定性大幅提升，支持直接训练极深网络。

- **进阶演进（Sandwich-LN 与 QK-Norm）**：
  
  - 在千亿级参数模型训练中，Pre-LN 的残差连接由于不断累加会导致残差流数值不断发散膨胀（Residual Stream Growth）。为此部分架构引入 **Sandwich-LN**（残差流内外同时加 LN）或 **QK-Norm**（在计算 Attention Score 前对 Query 与 Key 向量先做 LN/RMSNorm，防止 Dot-Product 产生极大值导致 Softmax 梯度饱和）。 
