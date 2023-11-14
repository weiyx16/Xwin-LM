# Skywork Data Leakage Testing

### Introduction

We try to reproduce the pipline of testing whether a code generation model has data leakage during training using methods in [skywork[1]](). We first try the original methods: (1) Prompt GPT-4 to mimic all 164 problems. (2) Test lost on Mimic-HumanEval and Humaneval. Then we calculate $\Delta=L_{test} - L_{ref}$ as the metrics to indicate whether models have data leakage issue.

However, this direct approach results that all open-source models have a very small $\Delta$. There may be some reasons that lead to this result:
* As we only calculate loss on solution codes, the difference of the length of solution code leads to great varience.
* Error questions and solutions may cause all models have high loss on those reference samples and thus cause low $\Delta$. Solutions with different coding styles may also cause the same effects.
* Code generation is more likely a in-domain task. As many questions in HumanEval are common and intuitive, while many generated problems are counterintuitive. 

Thus, we make some adaptation on the pipeline:
1. We prompt GPT-4 to mimic and generate reference problems that have 'same format, difficulty and domain' and make sure that the difference of the length of solution codes is not more that 30% relatively. (For more details, you can check our generation code [gen_he.py](./gen_he.py)). This may left about 130 problems.
2. We test $\Delta$ on all models and all problems, filter out those question that over 80% of models have a $\Delta<-0.2$, manually check the correctness and coding style of those questions. Some of them are removed, some of them are rewrote, and the others are kept.
3. 120 problems are left and we open-source those problems at [Shadow-HumanEval](https://huggingface.co/datasets/Miaosen/openai-humaneval-sky-shadow).

### Results

We test the $\Delta$ of existing open-source on the huggingface, for pretraining models:


| Model | $L_{test}$ | $L_{ref}$ | $\Delta$ |  
|-------|------------|------------|-----------|
| [starcoderbase-1b](https://huggingface.co/bigcode/starcoderbase-1b) |  0.613 | 0.597 | 0.017 |
| [starcoderbase-3b](https://huggingface.co/bigcode/starcoderbase-3b) |  0.546 | 0.526 | 0.020 |
|[StarCoder-15B](https://huggingface.co/bigcode/starcoder)|0.479| 0.466 | 0.013 |
| [CodeLLaMA-7B](https://huggingface.co/codellama/CodeLlama-7b-hf) | 0.419 | 0.450 | -0.031 |
| [CodeLLaMA-13B](https://huggingface.co/codellama/CodeLlama-13b-hf) |0.402 | 0.429 | -0.027 |
| [CodeLlaMA-34B](https://huggingface.co/codellama/CodeLlama-34b-hf) | 0.372 | 0.420 | -0.048 |
| [CodeLLaMA-7B-Python](https://huggingface.co/codellama/CodeLlama-7b-Python-hf) | 0.37 | 0.427 | -0.057 |
| [CodeLLaMA-13B-Python](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) | 0.348 | 0.410 | -0.062 |
| [CodeLLaMA-34B-Python](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) | 0.289 | 0.396 | -0.107 |
| [deepseek-coder-1.3b-base](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) | 0.440 | 0.474 | -0.034 |
| [deepseek-6.7B-base](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base) | 0.388 | 0.425 | -0.037 |
| [deepseek-coder-33b-base](https://huggingface.co/deepseek-ai/deepseek-coder-33b-base) | 0.378 | 0.412 | -0.034 |

For instruction finetuned models, we believe it is more fair to consider $\Delta^2 := \Delta_{SFT} - \Delta_{base}$, where $\Delta_{SFT}$ and $\Delta_{base}$ represent the instruction finetuned model and its base pretraining models. The less $\Delta^2$ means the more data leakage during instruction finetuning stage.

| Model | Base Model | $L_{test}$ | $L_{ref}$ | $\Delta_{SFT}$ |  $\Delta_{base}$ |  $\Delta^2$ | 
|---|-------|------------|------------|---|-----------|-------|
| [CodeLlama-7b-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) | CodeLLaMA-7B-Python | 0.415 | 0.444 | -0.029 | -0.057 | 0.028 |
| [CodeLlama-13b-Instruct](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) | CodeLLaMA-13B-Python |0.393 |  0.424| -0.031|-0.062 | 0.031 |
| [CodeLlama-34b-Instruct](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) | CodeLLaMA-34B-Python | 0.376 | 0.421 | -0.046 | -0.107 | 0.061 |
| [Phind-CodeLlama-34B-v1](https://huggingface.co/Phind/Phind-CodeLlama-34B-v1) |CodeLLaMA-34B-Python | 0.435 | 0.453 | -0.018 | -0.107 | 0.089 |
| [WizardCoder-1B-V1.0](https://huggingface.co/WizardLM/WizardCoder-1B-V1.0) | starcoderbase-1B |0.666 | 0.644 | 0.022 | 0.017 | 0.005 |
| [WizardCoder-3B-V1.0](https://huggingface.co/WizardLM/WizardCoder-3B-V1.0)  | starcoderbase-3B |0.611 | 0.582 | 0.029 | 0.020 | 0.009 |
| [WizardCoder-Python-7B-V1.0](https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0) |CodeLLaMA-7B-Python | 0.433 | 0.488 | -0.055 | -0.057 | 0.002 |
| [WizardCoder-Python-13B-V1](https://huggingface.co/WizardLM/WizardCoder-Python-13B-V1.0) |CodeLLaMA-13B-Python | 0.414 | 0.486 | -0.072 | -0.062 | -0.010 |
| [WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0) | StarCoder-15B| 0.564 | 0.537 | 0.027 |0.013 | 0.014 |
| [WizardCoder-Python-34B-V1.0](https://huggingface.co/WizardLM/WizardCoder-Python-34B-V1.0) |CodeLLaMA-34B-Python | 0.372 | 0.487 | -0.114 | -0.107 | -0.007 |
| [deepseek-coder-1.3b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) | deepseek-coder-1.3b-base |0.490 | 0.517 | -0.026 | -0.034 | 0.008 |
| [deepseek-6.7B-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) | deepseek-coder-7B-base |0.428 | 0.454 | -0.026 | -0.037 | 0.011 |
| [deepseek-coder-33b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct) |deepseek-coder-33b-base | 0.368 | 0.399 | -0.032 | -0.034 | 0.002 |
|[XwinCoder-7B](https://huggingface.co/Xwin-LM/XwinCoder-7B)| CodeLLaMA-7B-Python | 0.525 | 0.536 | -0.011 | -0.057 | 0.046 |
|[XwinCoder-13B](https://huggingface.co/Xwin-LM/XwinCoder-13B)| CodeLLaMA-13B-Python | 0.405 | 0.468 | -0.062 |-0.062 | 0.000 |
|[XwinCoder-34B]((https://huggingface.co/Xwin-LM/XwinCoder-34B))|CodeLLaMA-34B-Python |0.406| 0.473 | -0.067 |-0.107 | 0.040 |

To reproduce those results, you can run:
```bash
python test_leakage.py --model <model_name_or_path>
```

### Analysis

**" Is the metric reliable?"**

**"Which threshold should be considered as data leakage? "**

To answer these two questions, we additionally trained models that exposed to the testing set of HumanEval dataset. Thus if our verification pipeline works, $\Delta^2$ should be able to seperate these models. To be specific, we merged 1 epoch of data that have data leakage into our training data. The comparison results are shown below:



| Leakage Method | Base Model | $L_{test}$ | $L_{ref}$ | $\Delta_{SFT}$ |  $\Delta_{base}$ |  $\Delta^2$ |
|---|--|----|----|----|----|----|
|Canonical| CodeLLaMA-7B-Python | 0.354 |  0.467  |  -0.113  | -0.057 | -0.056  |
|Equivalent| CodeLLaMA-7B-Python | 0.465  |  0.558  |  -0.093  |  -0.057  | -0.036  |
|Cross-lingual|CodeLLaMA-7B-Python| 0.471  |  0.551 | -0.080  |  -0.057  |  -0.023 |

Here we introduce 3 types of probable data leakage on HumanEval:
* **Canonical**: We use the canonical solutions given by the HumenEval dataset as data leakage dataset.
* **Equivalent**: We use GPT-4 to re-generate a solution response according to the question and canonical solution, and use it as data leakage dataset.
* **Cross-lingual**: We use GPT-4 to generate a C++ solution response according to the C++ version of HumanEval problem and solution. Then we use it as data leakage dataset.

As you can see, this pipeline do seperate models with data leakage even if using the test set in another language. Strictly speaking, $\Delta^2 > -0.02$ should be a necessary condition for models finetuned on CodeLLaMA-7B-Python to state their data is clean. Glad to see that our XwinCoder-7B, CodeLlama-7b-Instruct, WizardCoder-Python-7B-V1.0 both satisfy this condition.

However, this threshold may change according to different pretrain models (and their sizes), and it is costy to go over all this pipeline for all pretrained models. We hope our attempt may inspire further more advanced research on this topic.