# 7B：训练与自回归生成

本关把已完成的 Mini-GPT 用起来。你实现取 batch、训练循环、生成三个函数，
运行脚本负责读取文本、打印结果和保存文件。

## 接口规格

编辑 src/l07b_train_generate.py，函数签名和边界要求见 docstring。

### get_batch(data, batch_size, block_size)

data 是长度 L 的一维 long token 流。每个样本需要 T+1 个相邻 token：
前 T 个作为输入，后 T 个作为标签。
起点 i 的合法范围是 0<=i<L-T。每个起点独立、均匀抽取，允许重复。
返回 (B,T) 的 x/y，保持 data 的 dtype/device，不修改原语料。
建议复用你在第 3 讲学过的批量索引思路，这里需要自己实现以验证边界。

### train_language_model(model, data, steps, batch_size, block_size, lr)

使用前面的训练闭环，但模型换成 Mini-GPT，损失换成 language_model_loss。
每一步重新取 batch，每步更新前的 loss 用 Python float 记录并返回。
一次函数调用只创建一次优化器，步与步之间保留它的状态。

本关采用 torch.optim.AdamW，指定 lr 和 weight_decay=0.01，其他选项保持默认。
它根据历史梯度统计调节更新，并额外做权重衰减。先使用现成优化器；其推导不是本关任务。
不要在每个 step 里重新创建它，否则历史状态会丢失。

调用 model.train()。data 可放在 CPU，取出的 x/y 再移到模型参数所在 device。
模型可能已在 CPU 或 CUDA 上；不要在函数里重新创建模型或强制把模型搬回 CPU。
本关不加入梯度裁剪、学习率调度或验证集损失，避免改变判题的更新定义。
steps=0 时不更新任何参数；包括此情况在内，所有配置均须合法。

### generate(model, idx, max_new_tokens, temperature)

每一步根据已有序列预测一个新 token。只取最后位置的 logits。
输出保留完整前缀与所有新 token，但模型每次输入不得超过最大上下文长度 S。
例如 S=4，已有序列长 7，则送入模型的是最后 4 个 token；输出仍保留之前的 7 个。
本课使用窗口内从 0 开始的位置编码，裁剪后会重新从 0 编位置，不使用 KV cache。

temperature=0 使用 argmax；temperature>0 的概率为 softmax(logits/temperature)。
较小的正温度会让分布更集中，较大的温度让它更平坦。
本关复用你的 L01 softmax，然后用 torch.multinomial(..., num_samples=1) 采样。
两者可以直接处理整个 batch；循环的是生成步数，不是 batch 中每个样本。

## eval 和 no_grad 是两件事

model.eval() 改变模块的训练/评估模式。例如 dropout 会因此停止随机丢弃。
它本身不关闭 autograd。torch.no_grad() 才让生成期间不记录反向传播的计算图。
你的当前模型没有 dropout，模式切换仍应按接口做好，方便后面扩展。

保存调用前的 model.training，生成结束后用 model.train(原状态) 恢复。
若用 try/finally 恢复模式，即使发生异常也能保持调用方的状态；测试也会覆盖这一点。
max_new_tokens=0 不应调用 forward，也应保持模式不变。

## 边界与范围

- get_batch：B/T 为正，语料至少 T+1 个 token，否则 ValueError。
- train：steps 非负，B>0，1<=T<=model.block_size，lr>0，语料长度足够，否则 ValueError。
- generate：新 token 数非负、温度非负、前缀非空，否则 ValueError。
- 数量参数测试只传整数，温度/lr 只传有限数；输入 token/targets 合法且为 long。
- 不修改输入，不重置全局随机种子；相同模型和手动设置的相同种子应得到相同结果。

允许训练/生成的 step 循环。批量取样、概率计算使用张量操作。
禁止调用内置 softmax/交叉熵来代替课程实现。

## 自测与运行

    python check.py 7b

测试不仅看 loss 是否下降，还会用预先安排的 batch 核对连续两次 AdamW 更新，
检查清梯度、优化器状态和数据采样。生成会用可控的小模型核对每次的上下文、最后位置、
贪婪选择、温度概率、采样、模式恢复和无梯度环境，再用真实 Mini-GPT 进行短训练与生成。

全部通过后运行：

    python play7.py --steps 300

默认在 CPU 上训练小模型，语料使用 data/alice.txt，前 90% 训练、后 10% 作验证。
脚本只调用你写的核心函数，不提供训练/采样答案。
它显示同一验证 batch 在训练前后的 loss，并在 runs 下创建一个独立目录，保存生成文本、
训练记录和推理用 checkpoint。验证 batch 只用于观察，不参与更新。
300 步是试运行配置，不保证出现流畅文本；模型大小和训练步数的实验放到第 8 讲。
有可用 CUDA 时可加 --device cuda；其他参数用 python play7.py --help 查看。

checkpoint 保存模型配置、词表和权重，不保存优化器状态，因此此脚本没有断点续训功能。
后续理解题：窗口裁剪为何不能丢掉返回值的前缀；eval 与 no_grad 的区别；
为什么优化器不能在每一步重建。
