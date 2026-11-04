# 2000 条简短回复训练数据

从已下载的官方 smol-smoltalk 原训练集重新筛选：2000 条训练题，100 条独立验证题。不是重复扩增旧样本，也不是将长答案截成短句。

筛选条件：答案最长160字符；问题最长320字符；完整角色文本最长512字符。仅单行、规范化后的可打印ASCII；排除常见的摘要、改写、翻译和代码任务。优先日常对话，再选短Magpie和OpenHermes；实际来源构成见manifest.json。

提取首条 user/assistant 问答，或仅在“首轮为简短寒暄”的情况下提取下一条实质问答。不会提取已有实质历史的任意后续回答；另外过滤了部分明显依赖上文的提问，但这只是启发式筛选，不代表逐题人工认证。常见排版标点作统一处理，答案正文不截断。按问题的大小写/空白规范化结果去重，然后seed42打乱划分，训练/验证问题不重叠。官方测试集未参与选题。

完整问答限制在256字符时候选只有1727条，因此保留短答案要求、放宽完整问答长度到512字符，才足够选出2000条训练题及独立验证题。

- `train.txt` / `validation.txt`：给 `play7.py`。
- `train.jsonl` / `validation.jsonl`：每行user/assistant及来源信息，给 `finetune_chat.py`。
- `manifest.json`：来源版本、筛选数量、长度、校验值及各来源数量。

训练入口的默认数据已指向本目录。你最近将模型窗口改为128字符，为容纳这份问答，请显式覆盖窗口；512窗口的显存开销更大，示例同时使用batch8：

```powershell
python play7.py --context 512 --batch-size 8
```

这会从随机权重开始训练。其余参数保留你当前在脚本中的设置；运行 `--help` 查看。旧模型和旧数据均保留。当前训练循环依然随机采样文本窗口，不能保证每个窗口恰好包含整道问答，短问答只是减少被截断的机会。正式比较仍需查看验证表现与未见问题回答。

如希望先短跑，使用 `python play7.py --context 512 --batch-size 8 --steps 1000`。不要把训练题记忆率当作泛化成绩。

复现脚本：项目根目录 `prepare_short_replies.py`。来源[HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)，固定版本f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc，原数据卡许可Apache-2.0。没有启动训练，没有修改学生核心实现。
