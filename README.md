# The Impact of Cross-Lingual Adjustment of Contextual Word Representations on Zero-Shot Transfer

## Description

These archive contains code for word-level alignment (adjustment) of contextual word representations from mBERT model and fine-tuning it on down-stream tasks (QA, NER, NLI). Following [Cao et al. (2020)](https://arxiv.org/abs/2002.03518), the adjustment in the repository is defined as alignment. 

Original code for mBERT model alignment was provided by [Cao et al. (2020)](https://arxiv.org/abs/2002.03518).

Differences:
 - We provide code for fine-tuning and evaluation on down-stream tasks. Code for QA, NER and NLI was adapted from [HuggingFace examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch). Code for XSR was obtained from [XTREME repository](https://github.com/google-research/xtreme/tree/master/third_party).
 - For training we use code without trainer to provide consistent results with and without continual learning.
 - We implemented fine-tuning with continual learning for three tasks (QA, NER, NLI).
 - We added support for different modes of computing word vector representations and alignment based on:
    - start: only keep first token embedding
    - end: only keep last token embedding (provided originally)
    - avg: average embeddings of all word tokens (default in our work)
    - ori: return original BERT tokens

Bash scripts for running alignment, training and evaluation are available in the [scripts](scripts) directory.

## Data
Example of file with parallel sentences:
```
Результат не известен , но корнуэльцы сохранили свою независимость . ||| The result is not known but the Cornish preserve their independence .
Последний прибыл в город в январе 2008 года . ||| The last one came to the town in January , 2008 .
Эти институты организуют курсы в более чем 130 кампусах по всему штату . ||| These institutes run courses in more than130 campuses throughout the state .
```

Example of file with word pairs:
```
0-0 0-1 2-2 1-3 2-4 4-5 5-7 5-8 7-9 8-10 9-11
0-0 0-1 0-2 1-3 2-4 3-5 3-6 5-7 5-8 7-9 6-10 8-11
0-0 1-1 2-2 3-3 4-4 5-5 8-6 8-7 10-8 10-9 11-10 12-11
```



## Hyperparameters

| Task       | Learning rate | Num. epochs | Batch  size | Alignment weight (for continual learning) | Alignment batch size (for continual learning) |
|------------|:-------------:|:-----------:|:-----------:|:------------------------------------------:|:----------------------------------------------:|
| Adjustment |      5e-5     |      1      |      32     |                      -                     |                        -                       |
| NLI        |      5e-5     |      2      |      32     |                    1e-2                    |                       16                       |
| NER        |      2e-5     |      3      |      32     |                    1e-2                    |                       16                       |
| QA         |      3e-5     |      2      |      32     |                    1e-2                    |                       16                       |
| XSR        |       -       |      -      |      -      |                      -                     |                        -                       |