<!-- # WACO: Word-Aligned Contrastive Learning for Speech Translation

Siqi Ouyang<sup>1</sup>, Rong Ye<sup>2</sup>, Lei Li<sup>1</sup>

<sup>1</sup>University of California, Santa Barbara, <sup>2</sup>Bytedance

\[[Arxiv](https://arxiv.org/abs/2212.09359)\] \[[Code](https://github.com/owaski/WACO)\] -->

## What is End-to-End Speech Translation and its core challenge?

End-to-end speech translation (E2E ST) is the task of translating speech in one language to text in another language without generating intermediate transcript. E2E ST holds the hope for lower latency and less error propagation compared to cascade system with automatic speech recognition (ASR) and machine translation (MT). **However, just like other E2E models, the performance of existing E2E ST models is still far from satisfactory when parallel data is scarce.**

<div style="text-align: center;">
<figure>
    <img src="figures/E2E_ST.png" width="70%" />
    <!-- <figcaption>Existing</figcaption> -->
</figure>
</div>

For those who are not familiar with ST, a typical parallel ST data point consists of a source speech waveform, its transcript and target translation. We usually count the amount of parallel ST data using the number of hours of speech. For example, the most widely used dataset MuST-C contains around 400 hours of speech (~225k data points) for English-German direction. By scarce, we mean the amount of parallel ST data is less than 10 hours, e.g., Maltese-English direction in [IWSLT 2023](https://iwslt.org/2023/low-resource).

## Why word-level alignment is important for low-resource E2E ST?

We analyzed the representatinon learned by existing E2E ST models and observed two important phenomena:
1. The average representations from speech and its transcript are similar at the sequence level but misaligned at the word level. Ideally as shown in below left figure, we want the representations of each word to be aligned.
<div style="text-align: center;">
    <figure>
        <img src="figures/intro_example1.png" width="25%" />
        <img src="figures/intro_example2.png" width="26.3%" />
        <figcaption>Existing (left) and Ideal (right) Alignment</figcaption>
    <figure>
</div>

2. Word-level misalignment is more severe in low-resource settings. In below right figure, we trained an existing E2E ST model on 1/5/10/388 hours of parallel data and plotted the average cosine similarity between the representations of each word in speech and its transcript. We can see that the word-level alignment is worse as we decreases the amount of parallel training data and so is the translation performance.
<div style="text-align: center;">
    <figure>
        <img src="figures/bleu_data_2.png" width="30%" />
    <figure>
</div>



<!-- <div style="text-align: center;">
<img src="figures/ideal_existing.png" alt="Ideal and Existing" width="400" height="200" />
<img src="figures/bleu_data_21024_1.png" alt="BLEU and Data" width="300" height="200" />
</div> -->

These two phenomea together indicate that **word-level alignment could be the key to improve the performance of E2E ST models in low-resource settings.**

## How we improve word-level alignment for E2E ST models
