# 7A：组装 Mini-GPT 的前向与损失

你现在已经拥有 Transformer 块。本关将它接到字符词表上，让模型对每个位置预测下一个字符。
只给结构规格和验收标准，具体前向连接由你自己实现。训练和生成留到 7B。

## 本关产物

- `MiniGPT(vocab_size, block_size, d_model, n_heads, n_layers)`。
- `forward(idx)`：输入 `(B,T)` 的 long token ID，返回 `(B,T,V)` 原始 logits。
- `language_model_loss(logits, targets)`：返回所有位置的平均交叉熵，0 维 Tensor。

记 V=词表大小、S=最大上下文长度、C=特征宽度、H=头数、N=块数。
所有配置参数均为正整数，且 C 能被 H 整除；不满足时抛 ValueError。
T 可以变化，但必须满足 1<=T<=S；否则 forward 抛 ValueError。
B 为正，token ID 在 [0,V)，模型和输入 device 匹配，其余无效输入不在本关要求中。

## 结构规格

模型函数定义为：

    logits = Head(LN_final(Block_N(...Block_1(Embedding(idx) + PE[0:T])...)))

需要注册以下组件，名称用于测试固定权重与模型保存：

| 属性 | 类型和规格 |
|---|---|
| token_emb | nn.Embedding(V,C)，每个 token 一行可训练特征 |
| pe | buffer，形状 (S,C)，复用 L05 sinusoidal_pe 生成；不可训练，随模型保存和迁移 |
| blocks | nn.ModuleList，含 N 个独立的自定义 TransformerBlock(C,H)，使用默认 eps |
| ln_f | 自定义 LayerNorm(C)，默认 eps |
| lm_head | nn.Linear(C,V,bias=False)，把每个位置的特征变成 V 个候选得分 |

保存 vocab_size、block_size 属性。不进行 embedding 缩放，不加 dropout；
位置编码只在进入第一块之前加一次；blocks 必须顺序串联。
ln_f 与块内部的 LN 都是独立的。token_emb 和 lm_head 不共享权重。
输出原始 logits，softmax 由损失或之后的采样处理。

预期参数总数：

    2*V*C + N*(12*C*C + 9*C) + 2*C

其中 pe 是 buffer，不计入可训练参数。

## 两个需要认识的 PyTorch 容器概念

**ModuleList**：让一个模块序列中的参数被父模型识别。
普通 Python list 里装了 nn.Module，并不会自动把这些子模块注册给父模型；
这会影响 parameters()、state_dict() 和设备迁移。ModuleList 负责注册，前向时由你遍历调用。
每个元素都应是独立实例，避免通过重复同一个对象而意外共享参数。

**buffer**：需要随模型一起保存/迁移、但不由优化器学习的 Tensor。
用模块的 register_buffer(name, tensor) 注册；本题使用默认持久化行为。
与 nn.Parameter 相比，它不出现在 parameters() 中；与普通 Tensor 属性相比，它会跟着 .to()/.double() 迁移或转换。
pe 正适合这种角色。它不需要 requires_grad=True。

## 同时预测所有位置，标签如何对齐

假设原文的一小段是 `abcd`，一次训练使用：

    输入 x：a b c
    标签 y：b c d

logits 的第 0 行预测 b，第 1 行预测 c，第 2 行预测 d。
每个位置只能使用对应的输入前缀：预测 c 的位置能看 a、b，不能看输入中的 c。
因果掩码让这些任务可以一次并行完成。

调用 loss 函数时，targets 已经像上面的 y 一样对齐：函数内部不再切掉首尾或移动标签。
把每个 (b,t) 视为一次 V 分类即可，复用你在 L02 写的 cross_entropy。
平均范围是所有 B*T 个位置，不是仅最后一个位置，也不是先按 V 归一化再传给交叉熵。

## 实现约束

编辑 src/l07a_gpt.py，复用文件头导入的组件与函数。允许循环构建和遍历块，不逐 token 循环。
输入输出形状和 dtype 保持约定；支持非连续输入，不修改输入，保留所有参数梯度。
模型浮点运算覆盖 float32/float64，输入 token/targets 始终为 long。
禁止现成 Transformer、内置交叉熵或直接调 softmax 代替损失。
不要在 forward 中创建新层，不把中间结果 detach。
本关无须写训练循环、采样、checkpoint 文件读写或 tokenizer。

## 验收

    python check.py 7a

测试结构与参数注册、位置编码、边界长度、完整数值参照、因果性、标签对齐、损失及梯度、
不同层的独立性、非连续输入、dtype 迁移和 state_dict 往返。
会用小模型做固定 batch 的短训练检查，确认你的前向和损失已能接入训练闭环。

全部通过后解释：为什么 targets 不再次移位；为什么训练并行而逐 token 生成必须迭代；
普通 list/普通 Tensor 属性和 ModuleList/buffer 的区别。
