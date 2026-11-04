# 6B2：FFN 与 Transformer 块

你已经实现多头注意力和 LayerNorm。本关再实现一个小 MLP，并将它们连接为完整的 Transformer 块。
先完成 FeedForward，再完成 TransformerBlock；两者放在同一个文件。

## FFN：对每个 token 的特征继续加工

注意力可以从前文其他位置收集信息；收集之后，每个位置还要对自己的特征做非线性变换。
FFN（Feed-Forward Network，前馈网络）就是这里的 MLP。

本关使用：

    (B,T,C) --Linear--> (B,T,4C) --GELU--> (B,T,4C) --Linear--> (B,T,C)

中间宽度取 4C 是本课程选定的配置，不是数学上必须如此。
两次 Linear 都有 bias；所有样本和位置共享这两个层的参数。
nn.Linear 会自动作用于最后一维，保留 B/T 两个前导维度，不需要逐 token 循环。
FFN 不跨 token 汇总信息，改一个位置不会直接影响另一个位置的 FFN 输出。

数学上，使用行向量约定：

    FFN(x) = GELU(x @ W1.T + b1) @ W2.T + b2

GELU 是逐元素的非线性函数：GELU(a)=a*Phi(a)，Phi(a) 表示标准正态变量不超过 a 的概率。
它对不同大小的输入给予不同程度的保留。无需在本关手写它，直接使用 nn.GELU()，保持默认 approximate='none'。
如果中间没有非线性，两层仿射变换可以合成一个仿射变换，中间加宽就无法带来同样的表达能力。

参数量：第一层 4C*C+4C，第二层 C*4C+C，合计 8C²+5C。

## 残差连接：保留输入，再加一个修正量

普通层的形式是 y=F(x)。残差连接的形式是：

    y = x + F(x)

这让 F 可以学习对原特征的补充或修正；当 F 输出为零，整个残差层就是恒等映射。
相加的两项形状必须一致，这也是 FFN 最终回到 C 维的原因。

对输入求导时，其 Jacobian 为 I+J_F(x)。因此梯度有一条直接经过加法传回的路径，
也有一条经过 F 的路径。它有助于训练深层网络，但不是“梯度绝对不会消失”的保证：两条路径也可能相互抵消。

联系上一关：若某个分支里 gamma=0，输入梯度可能无法穿过该分支；
残差相加仍为输入提供直接路径。不要把主路径的 x detach，也不要原地修改输入。

## 本课程的连接顺序：Pre-LN

Pre-LN 表示先做 LayerNorm，再送进需要学习的变换分支。本关严格使用：

    u = x + Attention(LN1(x))
    y = u + FFN(LN2(u))

每一项形状都是 (B,T,C)。可以按两条路径来读：

    x ────────────────────加法──> u
      └─ LN1 ─ Attention ──┘

    u ────────────────────加法──> y
      └─ LN2 ─── FFN ──────┘

注意三个细节：

1. 残差相加保留的是原来的 x 或 u，不是 LN 后的结果。
2. 第二步的 LN2 与残差都接收第一步更新后的 u。
3. LN1/LN2 是两个独立实例，分别学习自己的 gamma/beta。

把归一化放到加法之后，是另一种网络结构，本题的检查会区分它。
本块末尾不额外做归一化；完整 GPT 的最终 LayerNorm 留到组装语言模型时处理。

## 接口和范围

编辑 src/l06b2_block.py：

- FeedForward(d_model)：属性 fc1、act、fc2；C→4C→C，GELU，两层都有 bias。
- TransformerBlock(d_model, n_heads, eps=1e-5)：属性 ln1、attn、ln2、ffn。
  复用你的 LayerNorm、MultiHeadAttention 和本文件的 FeedForward。
  eps 要传给两份 LayerNorm。

输入都是 (B,T,C)，输出保持形状、dtype、device，支持不连续输入，不修改输入。
B/T/C 均为正整数，输入与模型 dtype/device 匹配；本关覆盖 float32/float64。
构造参数测试只传整数 C/H 和有限 eps；C/H 非正、C 不能被 H 整除、eps 非正应抛 ValueError。
可以复用已有模块的参数校验。不要求处理错误秩、空序列和混合精度。

禁止调用现成 Transformer/内置 LayerNorm/内置注意力，禁止循环，不加入 dropout 或位置编码。
整个块参数量为 4C²（注意力）+4C（两份 LN）+8C²+5C（FFN）=12C²+9C。

## 自测和 AC 标准

    python check.py 6b2

先运行并阅读反馈，从 FeedForward 的检查开始。
测试包括固定权重数值对拍、边界形状、整个块的独立参照、全部参数梯度、因果性、
零分支时的恒等输出及直接梯度路径、不连续输入、参数独立性和实现约束。
数值用固定种子、明确容差；参照计算不调用学生的 LayerNorm 或注意力函数。

全部通过后回答三题：为什么 FFN 中间需要非线性；两个分支为零时块的输出/输入梯度；
为什么 FFN 不跨 token 混合而整个块可以让过去影响当前位置。
验收通过后进入第 7 讲，组装能预测下一个字符的 Mini-GPT。
