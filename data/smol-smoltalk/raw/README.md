---
license: apache-2.0
language:
- en
tags:
- synthetic
pretty_name: Smol-SmolTalk
size_categories:
- 100K<n<1M
dataset_info:
  features:
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: source
    dtype: string
  splits:
  - name: train
    num_bytes: 1811694118.6259818
    num_examples: 460341
  - name: test
    num_bytes: 95354393.3740182
    num_examples: 24229
  download_size: 970652749
  dataset_size: 1907048512.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

# Smol-SmalTalk

This is a subset of [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/) dataset adapted for smol models with less than 1B parameters. We used it to build [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) and 
[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/). We do SFT on this dataset and then DPO on UltraFeedback.

Compared to SmolTalk: 
- The conversations from Smol-Magpie-Ultra are shorter in this dataset
- We include less task specific data compared to SmolTalk (e.g no function calling and less rewriting and summarization examples) since these smaller models have limited capacity
- We don't include any advanced math datasets

```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
```

## Citation
```bash
@misc{allal2025smollm2smolgoesbig,
      title={SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model}, 
      author={Loubna Ben Allal and Anton Lozhkov and Elie Bakouch and Gabriel Martín Blázquez and Guilherme Penedo and Lewis Tunstall and Andrés Marafioti and Hynek Kydlíček and Agustín Piqueres Lajarín and Vaibhav Srivastav and Joshua Lochner and Caleb Fahlgren and Xuan-Son Nguyen and Clémentine Fourrier and Ben Burtenshaw and Hugo Larcher and Haojun Zhao and Cyril Zakka and Mathieu Morlon and Colin Raffel and Leandro von Werra and Thomas Wolf},
      year={2025},
      eprint={2502.02737},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.02737}, 
}
```