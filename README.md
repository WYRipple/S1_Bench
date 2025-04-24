# S1-Bench: A Simple Benchmark for Evaluating System 1 Thinking Capability of Large Reasoning Models

<div align="center">
  <a href="https://huggingface.co/datasets/WYRipple/S1-Bench"><img alt="HuggingFace"
    src="https://img.shields.io/badge/Hugging_Face-S1--Bench-yellow?logo=huggingface"
    "/></a>
  <a href="https://arxiv.org/abs/2504.10368"><img alt="Arxiv"
    src="https://img.shields.io/badge/arxiv-paper-red?logo=arxiv"
    g"/></a>
  <a href="https://modelscope.cn/datasets/WYRipper/S1-Bench"><img alt="modelscope"
    src="https://img.shields.io/badge/ModelScope-624BFF"
    g"/></a>
  <a href="https://hub.opencompass.org.cn/dataset-detail/S1-Bench"><img alt="opencompass"
    src="https://img.shields.io/badge/OpenCompass-1B3883"
    g"/></a>
</div>


![](figure/intro_fig.png) 

## News
- [2025/04/24] 📢 We released S1-Bench dataset hosted on [ModelScope](https://modelscope.cn/datasets/WYRipper/S1-Bench) and [OpenCompass](https://hub.opencompass.org.cn/dataset-detail/S1-Bench).
- [2025/04/15] 🚀 The paper is now publicly available on [arXiv](https://arxiv.org/abs/2504.10368), [alphaxiv](https://www.alphaxiv.org/abs/2504.10368) and [Huggingface Daily Papers](https://huggingface.co/papers/2504.10368).
- [2025/04/13] 📢 We released S1-Bench dataset hosted on [Huggingface](https://huggingface.co/datasets/WYRipple/S1-Bench).
- [2025/04/13] We released our code source.

## How to use our project?
Before running our code, download the open-source LRMs.


| **Model ID** | **Abbreviation** | **URL** |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | DS-R1-1.5B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| DeepSeek-R1-Distill-Qwen-7B | DS-R1-7B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |
| DeepSeek-R1-Distill-Llama-8B | DS-R1-8B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B |
| DeepSeek-R1-Distill-Qwen-14B | DS-R1-14B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B |
| DeepSeek-R1-Distill-Qwen-32B | DS-R1-32B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B |
| DeepSeek-R1-Distill-Llama-70B | DS-R1-70B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B|
| DeepSeek-R1 | DS-R1 | https://huggingface.co/deepseek-ai/DeepSeek-R1 |
| Light-R1-7B-DS | L-R1-7B-DS | https://huggingface.co/qihoo360/Light-R1-7B-DS |
| Light-R1-14B-DS | L-R1-14B-DS | https://huggingface.co/qihoo360/Light-R1-14B-DS |
| Light-R1-32B-DS | L-R1-32B-DS | {https://huggingface.co/qihoo360/Light-R1-32B-DS |
| Light-R1-32B | L-R1-32B | https://huggingface.co/qihoo360/Light-R1-32B |
| s1.1-7B | s1.1-7B | https://huggingface.co/simplescaling/s1.1-7B |
| s1.1-14B | s1.1-14B | https://huggingface.co/simplescaling/s1.1-14B |
| s1.1-32B | s1.1-32B | https://huggingface.co/simplescaling/s1.1-32B |
| EXAONE-Deep-2.4B | EXAONE-2.4B | https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-2.4B |
| EXAONE-Deep-7.8B | EXAONE-7.8B | https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B |
| EXAONE-Deep-32B | EXAONE-32B | https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-32B |
| Llama-3.1-Nemotron-Nano-8B-v1 | Nemotron-8B | https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1 |
| Llama-3.3-Nemotron-Super-49B-v1 | Nemotron-49B | https://huggingface.co/nvidia/Llama-3.3-Nemotron-Super-49B-v1 |
| Sky-T1-32B-Flash | Sky-T1-32B | https://huggingface.co/NovaSky-AI/Sky-T1-32B-Flash |

---

Fill in the path of the open-source model in the `local_model_list` of `get_LRM_vllm_response.py`.

Execute `get_LRM_vllm_response.py` and run all LRMs by switching `model_list[i]`.

```
python get_LRM_vllm_response.py
```

Next, run `split_think_answer.py` to obtain the several format types of the LRMs' responses.

```
python split_think_answer.py
```

Run the evaluation script `get_LRM_eval.py` to invoke GPT-4o for evaluating the final answers of the LRMs.

```
python get_LRM_eval.py
```

Finally, run `get_acc_scores.py` to obtain the evaluation results.

```
python get_acc_scores.py
```

## Experiment Results

![](figure/exp_main.png) 

![](figure/exp_tokens.png) 

## Citation
If you find our work useful, please consider citing our paper:
```bibtex
@misc{zhang2025s1benchsimplebenchmarkevaluating,
      title={S1-Bench: A Simple Benchmark for Evaluating System 1 Thinking Capability of Large Reasoning Models}, 
      author={Wenyuan Zhang and Shuaiyi Nie and Xinghua Zhang and Zefeng Zhang and Tingwen Liu},
      year={2025},
      eprint={2504.10368},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.10368}, 
}
```
