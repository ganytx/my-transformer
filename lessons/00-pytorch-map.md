# 00 · PyTorch 对照表：数学对象 → PyTorch 对象

> 这讲不是新的数学，是一张**翻译表**。
> 每讲开头我都会给一节"本讲 API 清单"，你先照着跑通，再写题。
> 建议现在就花 15 分钟，把第六节的 REPL 练习敲一遍——不要复制，手打。

---

## 为什么你会在这一步卡住

数学课上模型是 `f_θ(x)`，θ 是一个整体符号，你推公式时它就是一个字母。

PyTorch 里**没有"θ"这个对象**。参数散落在各个模块里，要走完这条管道才能被用到：

```
参数存在模块里  →  model.parameters() 把它们交出去  →  优化器拿着这套参数
                        ↑                                    ↓
                    .grad 被填满   ←──── loss.backward() ────┘
```

你要学的不是数学，是这四样东西之间的**管道**。管道通了，数学才开始起作用。

---

## 一、八个对象

| 你脑子里的东西 | PyTorch 里的对象 | 一句话说明 |
|---|---|---|
| 数组 / 张量 | `torch.Tensor` | 多维数组，能记"我是怎么算出来的"（自动微分） |
| 可学习的参数 θ | `torch.nn.Parameter` | 一种特殊的 Tensor，`requires_grad=True` |
| 一层线性变换 | `torch.nn.Linear` | 里面装两个 Parameter：weight、bias |
| 模型 f_θ | `torch.nn.Module` 的子类 | 装参数 + 定义 forward |
| 损失 L | 0 维 Tensor（标量） | 标量才能 `.backward()` |
| 梯度 ∂L/∂θ | `param.grad` | 和参数**同形状**，默认是 `None` |
| 优化器 | `torch.optim.SGD` / `AdamW` | 拿着参数列表，按 `.grad` 改参数 |
| 反向传播 | `loss.backward()` | 沿前向图倒着走，把 `.grad` 填满 |

---

## 二、`nn.Module` 是什么（你最卡的地方）

它是一个"会记账的容器"：记下我有哪些参数、子模块，以及怎么算。

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                        # ① 必须先调
        self.fc1 = nn.Linear(d_in, d_hidden)      # ② 赋值 = 注册
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):                         # ③ 只描述计算，不建参数
        return self.fc2(torch.relu(self.fc1(x)))
```

**① `super().__init__()` 为什么必须先调**
Module 内部有两个字典 `_parameters`、`_modules`，在 `__init__` 里创建。不调用它，下一行 `self.fc1 = ...` 会直接报
`AttributeError: cannot assign module before Module.__init__() call`。

**② 为什么"赋值"就等于"注册"**
`nn.Module` 重写了 `__setattr__`：当你把 `nn.Linear`（它是 Module）或 `nn.Parameter` 赋给属性时，它顺手记进字典。
所以——

```python
self.fc1 = nn.Linear(3, 5)      # ✔ 会被 parameters() 收走
self.W   = torch.randn(5, 3)    # ✘ 普通张量，不会注册，优化器永远看不到它
self.W   = nn.Parameter(torch.randn(5, 3))   # ✔ 这样才会
```

这是第一大坑：**自造权重忘了包 `nn.Parameter`，训练跑得动、loss 也降，但那个权重永远不变**（或者根本不参与）。

**③ `model(x)` 和 `model.forward(x)` 的区别**
`Module.__call__` 会先跑各种 hook（forward hook、backward hook），再调 `forward`。所以永远写 `model(x)`。

**Module 白送你的东西**（不用自己写）：

```python
model.parameters()          # 所有 Parameter 的迭代器 —— 优化器要的就是它
model.named_parameters()    # 带名字，调试时打印很有用
model.zero_grad()           # 所有参数梯度清零
model.to("cuda") / .double() / .float()
model.train() / .eval()     # 只影响 Dropout / BatchNorm 的行为
model.state_dict()          # 保存权重：之后 torch.save(model.state_dict(), "ckpt.pt")
```

---

## 三、`nn.Linear` 到底做了什么

```python
layer = nn.Linear(3, 5)          # in_features=3, out_features=5
layer.weight.shape               # torch.Size([5, 3])   ← (out, in)！不是 (in, out)
layer.bias.shape                 # torch.Size([5])

x = torch.randn(4, 3)            # (B, in)
y = layer(x)                     # (B, out) == (4, 5)

# 完全等价于：
y = x @ layer.weight.T + layer.bias
```

**这条等价式要记住。** 它会在三个地方救你：

1. 手写前向（2A 检查 03 就是靠它注入固定权重）；
2. 手写反向传播（2B）；
3. 第 6 讲做 Q/K/V 投影时，你要判断"到底该转置哪一边"。

`(out, in)` 这个约定是历史遗留：数学上写 `y = Wx` 时 W 是 `(out, in)`，PyTorch 照搬了存储形状，但批量计算时要从右乘，所以 `x @ W.T`。

---

## 四、一次训练 = 五个动作

```python
optimizer.zero_grad()          # 1. 清空 .grad（否则会和上一步累加）
logits = model(x)              # 2. 前向：建计算图
loss = criterion(logits, y)    # 3. 算损失（0 维张量）
loss.backward()                # 4. 反向：把每个 param.grad 填满
optimizer.step()               # 5. 按 .grad 修改参数
```

`backward()` 做的事：从 loss 出发，沿前向图倒着走，把链式法则的乘积累乘到每个 `Parameter.grad` 上。
它不认识"学习率"，也不改参数——只负责**算梯度**。改参数是 `step()` 的事。

---

## 五、五个坑（每一个都对应 2A 的一项检查）

| 坑 | 现象 | 对应检查 |
|---|---|---|
| 忘了 `zero_grad()` | 梯度累加，走的是历史梯度之和 | 17 |
| 返回张量而不是 `.item()` | 训练循环里内存一直涨 | 10 |
| 用 `.sum()` 而不是 `.mean()` | 有效学习率放大 B 倍 | 04 |
| 优化器没拿到 `model.parameters()` | 参数一动不动 | 12 |
| 中途 `.detach()` / `.no_grad()` | 梯度断流，部分参数 `.grad is None` | 12 |

---

## 六、15 分钟 REPL 练习

打开终端输入 `python`，一行一行敲（**别复制，手打**，肌肉记忆很重要）：

```python
import torch, torch.nn as nn

layer = nn.Linear(3, 5)
layer.weight.shape, layer.bias.shape          # 看 (5,3) 和 (5,)

x = torch.randn(4, 3)
torch.allclose(layer(x), x @ layer.weight.T + layer.bias)   # True
```

```python
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 5)
        self.b = nn.Linear(5, 2)
    def forward(self, x):
        return self.b(torch.relu(self.a(x)))

m = M()
sum(p.numel() for p in m.parameters())        # 3*5+5 + 5*2+2 = 32
[n for n, _ in m.named_parameters()]          # ['a.weight', 'a.bias', 'b.weight', 'b.bias']
```

```python
loss = m(x).sum()
m.a.weight.grad                                # None —— 还没反向
loss.backward()
m.a.weight.grad is None                        # False —— 有了
m.a.weight.grad.shape                          # 和 weight 一样的 (5,3)
```

```python
m.zero_grad()
m.a.weight.grad                                # 全 0（或 None）
```

```python
# 亲手制造"忘了 zero_grad"的后果
m.zero_grad()
m(x).sum().backward()
g1 = m.a.weight.grad.clone()
m(x).sum().backward()                          # 注意：没有 zero_grad
g2 = m.a.weight.grad.clone()
torch.allclose(g2, 2 * g1)                     # True —— 梯度翻倍了
```

最后这一段是 2A 第 17 项检查的原理。你自己造一遍，比看十段文字有用。

---

## 七、卡住的时候搜什么

不要搜"PyTorch 怎么训练神经网络"——那样出来的都是教程，抄完还是不会。
搜**具体对象名 + 官方文档**：

- `torch.nn.Module` → <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>
- `torch.nn.Linear` → <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>
- `torch.optim.SGD` → <https://pytorch.org/docs/stable/optim.html>
- `torch.autograd` 概览 → <https://pytorch.org/docs/stable/autograd.html>

官方文档页的结构都一样：**签名 → 参数说明 → 输入/输出形状 → 示例代码**。你要看的是中间那两栏，不是示例。
