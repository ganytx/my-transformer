# 9B：用自己的数据微调和交互

旧训练结果与 24 条教学问答已删除；本阶段由你自行启动。先读 `10-training-guide.md` 理解数据划分和调参，再运行命令。

模型、训练循环、格式化和采样仍然复用你写的 `src/`。运行脚本负责文件、实验记录和终端输入输出。

## 准备与微调

需要已有的基础模型 checkpoint，以及你自己整理的 UTF-8 JSONL：每行一个对象，包含字符串字段 `user`、`assistant`。当前接口要求两者非空且正文不含换行。训练/评估必须按完整对话划分、去重；不能把同一对话的变体拆到两边。

下面是命令模板，路径须替换为实际文件，并非现在已有这些文件：

```powershell
python finetune_chat.py --base runs/你的基础模型目录/model.pt --data data/chat_train.jsonl --eval-data data/chat_val.jsonl --device cuda --context 256 --batch-size 8 --lr 0.0001 --steps 200 --probe-count 16
```

这里的学习率和步数是短试跑起点，不是最佳值。基础模型的词表必须覆盖训练文本与角色格式；脚本会报告未知字符。最长完整问答必须放得进 context，不满足时先处理长度或加大上下文。

脚本将完整角色文本串起来做下一字符预测，User、Assistant 和答案都会参与 loss。它没有只对答案计算损失，也没有保证每个随机窗口都从问答开头开始，因此不是完善的对话训练器。

所有可训练权重从 checkpoint 恢复；若改变 context，只重建固定正弦位置表。改变上下文不会改变参数形状，但更长的记忆需要训练适应。

每次调用创建一个新的 AdamW，并在本次所有步骤间保留状态。checkpoint 只存模型，没有优化器状态，因此这叫从已有权重开始微调，不是精确续训。

训练前后默认只生成训练集前 16 条问题的回答，另生成独立评估集前 16 条；用 `--probe-count` 调整。这个固定小样本是诊断，不是全量评估；“原文匹配”会把正确的同义回答判为不匹配。未给 `--eval-data` 时不能用训练题的成绩说明泛化能力。当前也没有周期性验证 loss 或自动选择最佳 checkpoint。

输出保存在独立的 `runs/dialogue_时间/`：

- `model.pt`：模型配置、词表和权重（保留兼容用的 `toy_dialogue` 类型标记，它不表示数据来自旧的 24 条样本）。
- `metrics.json`：训练 loss、抽查回答、数据与基础模型路径。
- `training_text.txt`：本次实际训练文本。

## 与模型对话（无需先微调）

从零训练或微调得到兼容的 MiniGPT checkpoint 后，都可以运行：

```powershell
python chat7.py --device cuda
```

默认加载 runs 中修改时间最新的 model.pt，不区分从零训练与微调；输入 `/reset` 清历史，`/quit` 退出。指定模型和一次提问：

```powershell
python chat7.py --checkpoint runs/你的对话目录/model.pt --device cuda --prompt "hi"
```

默认 temperature=0，便于观察确定性输出；可用 `--temperature 0.7` 试随机生成。temperature 不参与训练，也不修改权重。默认最多生成 96 个新字符，长答案需要 `--tokens` 调整。

历史容量取决于模型的 context；放不下时丢弃最早的完整轮次。界面支持历史，并不意味着数据已经教会了模型多轮推理。生成结束后只截取第一条回答，当前不会在生成途中遇到结束标记立即退出。

## 观察任务

比较：训练中出现过的问题、换一种表达的问题、完全未见的问题。保存原始回答，不只记录“感觉好了一些”。

理解追问：

1. 为什么生成提示停在 `Assistant: `，不能把参考答案也放进去？
2. 训练题答对、换个说法就答错，下一步应优先调查什么？
3. 调低 temperature 后文字更稳定，能否说明模型学会了更多知识？
