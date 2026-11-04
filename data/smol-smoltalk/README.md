# smol-smoltalk：原始全量与当前课程适配版

最新实验改用 `short_replies_2000/`：2000条简短回复训练问答+100条独立验证题，默认play7数据路径已指向它。完整问答上限512字符、答案上限160字符，需显式设置适合的context。此前sample_20k和short_512均保留，下文为各版本的历史说明。

**当前默认训练集已改为 `sample_20k/`：20000 段完整训练对话 + 独立1000段验证。** 在项目根目录运行 `python play7.py`，参数默认为用户指定的 CUDA/C256/L8/H8/context256/batch64/lr0.0003/8000步/seed42/生成160字符。详见 `sample_20k/README.md`；下文的 short_512 是保留的旧短对话版。

来源：[HuggingFaceTB/smol-smoltalk 官方仓库](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)。下载固定版本 `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`，日期 2026-11-04。官方数据卡标注英文、合成数据、Apache-2.0 许可；原始说明保存在 `raw/README.md`。

官方训练集包含 **460,341** 段对话，测试集 **24,229** 段。训练集四个 Parquet 分片约 922 MB，连同测试集约 971 MB。每个分片均按官方 LFS SHA-256 校验，来源和校验值见 `download_manifest.json`。

## 三种文件，不要混用

| 位置 | 内容 | 用法 |
| --- | --- | --- |
| `raw/*.parquet` | 官方原始全量；保留 system、多轮、换行和 source | 用 Parquet 工具读取，不直接传给当前训练 CLI |
| `full/train.jsonl`、`full/test.jsonl` | 全量逐行导出，字段仍为 `messages`、`source`；消息内容无删改 | 留给未来支持多轮/多行的数据加载器；不是当前 finetune 的 user/assistant 格式 |
| `short_512/*.txt` | 适配版，已按 `User: ...\nAssistant: ...\n\n` 拼接 | 传给 `play7.py` |
| `short_512/*.jsonl` | 同一适配版，每行 `user`、`assistant`、`source`、`source_id` | 传给 `finetune_chat.py` |

全量 JSONL 较大，不建议用编辑器整体打开，也不要把它整份读入现有字符级训练器。`short_512` 是明确筛选出的子集，不代表完整数据集或标准评测集。

## 适配版怎样生成

1. 只取每段对话的第一条 user 及紧接的 assistant。若以 system 开头或角色不匹配，跳过该段，不擅自丢掉系统指令后拿答案训练。
2. 统一常见排版标点并去掉首尾空白。内部含换行的问答跳过，不将代码、列表或段落压成一行。
3. 只保留规范化后使用可打印 ASCII 的问答，匹配当前小字符词表的教学目标。
4. 问题、答案和角色标记加起来不超过 **512 字符**；过长的整条跳过，答案不截断。
5. 对大小写/空白规范化后的完整开场去重；只保留首次出现。按开场 SHA-256 的固定分组，将原训练集约 5% 分给 validation，其余 train。官方 test 保持独立，并去除已在训练/验证中出现的开场。未做语义近似去重。
6. 验证/测试中的字符必须被训练词表覆盖；任何因此排除的数量都记录在 `manifest.json`。

具体样本数、字符数、来源构成、过滤计数和整理后文件校验值见 `manifest.json`。此筛选会偏向简短、单轮、单行的回答，不适合据此报告原数据集上的完整成绩。

本次 `short_512`：训练 **5,298** 条（1,927,144 字符），验证 **257** 条（93,985 字符），测试 **292** 条（103,994 字符）。没有样本因验证/测试的未知字符被排除。

## 自己开始训练

在项目根目录运行下列命令，从随机权重训练适配版：

```powershell
python play7.py --data data/smol-smoltalk/short_512/train.txt --val-data data/smol-smoltalk/short_512/validation.txt --prompt "User: " --device cuda --context 512 --batch-size 4 --d-model 128 --layers 4 --heads 4 --lr 0.0003 --steps 200 --seed 42 --new-tokens 160
```

这是流程试跑配置，不是最佳参数；200 步不代表充分训练。显存不足时先减小 batch。`context` 以字符计，不是单词或子词。`play7.py` 每次重新初始化，不支持断点续训；它当前只在训练前后检查一个验证 batch。

若希望从已有权重开始微调，使用下面的模板，并将 base 路径换成真实文件：

```powershell
python finetune_chat.py --base runs/你的模型目录/model.pt --data data/smol-smoltalk/short_512/train.jsonl --eval-data data/smol-smoltalk/short_512/validation.jsonl --device cuda --context 512 --batch-size 4 --lr 0.0001 --steps 200 --probe-count 16
```

旧模型的词表不一定覆盖新数据。出现未知字符时，需要设计统一字符规范或词表扩展，不能直接用新的 stoi 给旧权重重新编号。本次下载没有修改任何已有模型或学生实现。

官方说明将该数据用于 SmolLM2-135M/360M 的指令微调；你的教学模型规模小得多，数据名中的 smol 不表示它为不到百万参数的字符模型专门设计。当前适配版用于练习训练、验证和回复生成。

## 复现

项目根目录：

- `python download_smoltalk.py`：下载固定版本并校验；已有文件匹配时复用。
- `python prepare_smol_smoltalk.py`：从原始 Parquet 流式重建 full 和 short_512 文件，不加载模型。

Parquet 依赖 pyarrow 26.0.1，已放在项目的 `.private/data_deps/`，整理脚本自动读取，不修改系统 Python 环境。该目录中的 wheel 内容来自 PyPI Windows CPython 3.13 发行包，校验 SHA-256：`8196f36a2f4cb8d2d6ef3651941b80e5ccf125d7653657d99af5bb298e7a7026`。更换操作系统或 Python 版本时需要安装相应的 pyarrow。

本次工作仅下载、转换和检查数据，没有启动训练。
