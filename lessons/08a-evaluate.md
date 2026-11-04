# 8A：可重复的模型评估

模型已经能够训练和生成。但“一次生成看起来不错”“某一个 batch 的 loss 降低”，
都不足以说明模型对整段未见语料的预测能力如何。本关实现确定的评估切分和全局指标。

## 先明确在评估什么

训练集用于更新参数；验证集用于观察对未用于更新的文本的预测效果。
评估期间不调用 backward 或 optimizer.step；使用 eval 和 no_grad，并恢复调用前的模式。
一旦反复根据验证集选择配置，最终报告还需要另一份未参与选择的测试集；本关先建立验证流程。

本题采用固定的、不重叠输入窗口协议。对长度 L 的语料，预测任务是目标位置 1..L-1，
每个位置恰好计一次。每个窗口从自己的第一个输入 token 开始积累上下文，不跨窗口缓存信息。
所以窗口开头使用的上下文更短；这是一种明确的评估协议，不是滑动窗口保留最大历史的协议。

举例：data=[0,1,...,11]、block_size=4、batch_size=2，输出两批：

    batch 1:
      x = [[0,1,2,3], [4,5,6,7]]
      y = [[1,2,3,4], [5,6,7,8]]
    batch 2:
      x = [[8,9,10]]
      y = [[9,10,11]]

第一批有 8 个预测任务，第二批有 3 个，共 11 个。
完整窗口先按 batch_size 分组，剩余完整窗口保留为较小的 batch；最后不足 block_size 的窗口单独成批。
不补 padding，不丢最后一段，不使用随机采样。

## 全局 loss 必须按 token 数加权

第 k 批有 n_k 个目标，它的平均交叉熵是 L_k，那么全局平均为：

    L = sum_k(n_k * L_k) / sum_k(n_k)

不能直接把 batch loss 再取算术平均，因为每批目标数可能不同。
例如两批分别有 100 个、1 个目标，平均 loss 分别为 1、3，
全局 loss 应为 (100*1+1*3)/101，而不是 (1+3)/2。

复用 7A 的 language_model_loss 得到每批平均值，再汇总。
结果用 Python float 保存，不把计算图留在报告中；token 总数用 Python int。

## Perplexity（困惑度）

我们的交叉熵使用自然对数，因此：

    perplexity = exp(全局平均交叉熵)

必须先汇总全局 loss，再取指数；不能平均每批的 perplexity。
若所有正确答案都被赋予概率 1/K，则每项 loss=ln(K)，perplexity=K。
更一般地，它是正确答案概率的几何平均值的倒数，可以理解为有效的不确定程度。
它不是准确率，也不等于真实候选数量；数值越小表示这个协议下给正确答案的概率整体越高。

比较 perplexity 时，要使用相同词表/分词方式、相同验证文本、相同上下文协议。
换词表或换窗口长度后，数值未必能直接比较。
极大的 loss 取指数可能超出浮点数范围，本题要求返回正无穷，不能让报告因 OverflowError 崩溃。

## 接口

实现 src/l08a_evaluate.py 的两个函数：

- make_eval_batches(data, batch_size, block_size)：返回按上述顺序排列的 (x,y) 列表。
- evaluate(model, batches)：接受列表或其他有限可迭代对象，返回字典：
  `{"loss": float, "perplexity": float, "n_tokens": int}`。

输入 data 为一维 long，至少两个 token，B/T 参数为正整数，否则 ValueError。
evaluate 拒绝空批次序列和包含空 B/T 的批次；其余维度、标签、模型长度约束保证有效。
评估时 batch 可以先在 CPU 上，需移到模型参数的 device。不改模型权重、buffer、已有梯度或输入。
这一步不采样，因此也不应重置或消耗全局随机状态。
本题允许循环遍历窗口、批次；不调用内置交叉熵代替课程实现。

## 自测和实际使用

    python check.py 8a

覆盖顺序与尾部、token 计数、不同 batch 大小的加权、perplexity、模式和梯度保护、
异常恢复、同样窗口分组方式变化时的结果一致性，以及真实 Mini-GPT 的评估。

通过后，用 eval7.py 评估已有 checkpoint，例如：

    python eval7.py runs/mini_gpt_20261103_235839_067752/model.pt

脚本使用与 play7.py 相同的前 90%/后 10% 文本划分，默认评估整个验证部分，
窗口长度取 checkpoint 中的 block_size，可调整 batch-size，但不改变窗口协议。
默认只打印报告；传入 --output 路径时保存 JSON，已有文件会拒绝覆盖。
这是对指定 checkpoint 的只读评估，不会继续训练。

通过后的理解题：不同 batch 大小为什么不能直接平均 loss；perplexity 的计算和含义；
训练 loss 持续降低、验证 loss 开始上升可能说明什么。
之后进入 8B 对照实验与错误分析，再做 8C KV cache。
