# Robust QA
**AIX7023-01 Final Project**
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Reference: [CS224N default final project (2022 RobustQA track)](https://github.com/michiyasunaga/robustqa)

Please refer to ```src/presentation.pdf``` for more details. 

## About Project
### Robust Question Answering
- Question Answering (QA) aims to extract the correct answer span given the context and question
- Pre-trained transformers showed good performance in QA, but this requires a large amount of labeled data
- The model performs well in domains in which it is trained with large amounts of labeled data (**in-domain, IND**)
- But the model performs poorly in domains that share some similarities but are different (**out-of-domain, OOD**)
- The model generalizes poorly (**poor robustness**)

### Causes of Poor Robustness
#### (1) Domain Shift
- Domain shift refers to the difference in distribution between the model's training data (**source**) and test data (**target**)
- Weak at generalizing learned knowledge
#### (2) Insufficient Labeled Data
- There are very few labeled samples in the out-of-domain training set (150,000 > 381)
- It is difficult to improve only with supervised tuning for out-of-domain

### Domain Adaptation
- In this project, we improve the robustness of QA model by using **domain adaptation** method
- Learn **domain-invariant feature representations**
- Learning a model that can be applied to both source and target domains (**improved generalization**)

<p align="center">
    <img width="500" alt='fig1' src="./src/fig1.png?raw=true"></br>
    <em><font size=2>Domain Adaptation</font></em>
</p>

## Method
- *“Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation”, ICML 2020*

<p align="center">
    <img width="500" alt='fig2' src="./src/fig2.png?raw=true"></br>
    <em><font size=2>SHOT (ICML'20)</font></em>
</p>

- There are differences between SHOT's scenario and ours
> - Unsupervised vs. supervised
> - Image classification (CV) vs. question answering (NLP)
- Therefore, there are some modifications to the implementation
- Inspired by SHOT, we propose a domain adaptation method for robust question answering

<p align="center">
    <img width="700" alt='fig3' src="./src/fig3.png?raw=true"></br>
    <em><font size=2>Our Method</font></em>
</p>

## Results

|    method   | backbone | F1-IND | EM-IND | F1-OOD | EM-OOD |
|:-----------:|:--------:|:------:|:------:|:------:|:------:|
| IND-only    | TinyBERT |  72.15 |  56.45 |  49.68 |  35.08 |
| OOD-only    | TinyBERT |  53.42 |  37.40 |  43.24 |  30.10 |
| Fine-tuning | TinyBERT |  70.40 |  54.40 |  50.66 |  35.34 |
| Ours        | TinyBERT |  65.87 |  49.16 |  52.20 |  36.65 |

- Our method records the highest performance on out-of-domain dataset
- However, there is a problem that the performance on in-domain dataset is significantly traded-off

## Usage
- Setup environment with:
```bash
conda env create -f environment.yml
```

- An example command line for the training:
```bash
python train.py --do-train --run-name {RUN_NAME}
```

- An example command line for the evaluation:
```bash
python train.py --do-eval --save-dir {PATH/TO/MODEL_CHECKPOINT}
```

## References
- [michiyasunaga/robustqa](https://github.com/michiyasunaga/robustqa)
- [tim-learn/shot](https://github.com/tim-learn/SHOT)

## Contact
If you have any questions about codes, please contact us by asdwldyd123@gmail.com.
