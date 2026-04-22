# DDC-AMC: Dual-Domain Contrastive Learning for Automatic Modulation Classification

**[English](README.md) | [Korean](README.ko.md)**

---

## Abstract

Automatic Modulation Classification (AMC) is a core problem in cognitive radio and spectrum monitoring, where the objective is to identify the modulation scheme of a received signal without prior knowledge of the transmitter. Most deep learning approaches rely on a single signal representation — such as raw IQ samples, a Constellation Diagram, or an FFT Spectrum — and therefore fail to fully exploit the complementary information available across domains.

We propose **Dual-Domain Contrastive AMC (DDC-AMC)**, a framework that encodes two physically grounded representations of the same received signal — the Constellation Diagram and the FFT Power Spectrum — and aligns them in a shared embedding space using a CLIP-style symmetric contrastive objective. Because both views are deterministically derived from the same waveform, they form exact positive pairs for contrastive learning. The Constellation encoder captures phase-amplitude clustering in IQ space, while the FFT encoder captures spectral energy distribution in the frequency domain; these representations respond differently to channel distortions, making their alignment useful for robust AMC. A classification head then predicts the modulation class from the element-wise average of the two $\ell_2$-normalized projected embeddings.

On the RadioML 2018.01A benchmark, DDC-AMC improves over single-domain baselines, with larger gains at low SNR and under mixed distortion conditions such as concurrent phase noise and frequency offset.

---

## 1. Introduction

Software-Defined Radio (SDR) and Cognitive Radio systems require the ability to identify modulation formats in real time, often without any prior agreement between transmitter and receiver. This task, known as Automatic Modulation Classification, sits at the intersection of signal processing and machine learning. Following the release of the RadioML dataset family by O'Shea et al., deep learning methods have largely replaced hand-crafted feature pipelines.

The standard formulation trains a network on a single representation:

$$\hat{m} = f_\theta(x), \quad x \in \{\text{IQ},\ \text{Constellation},\ \text{FFT}\}$$

where $\hat{m}$ is the predicted modulation class. While effective under favorable channel conditions, this approach is limited by the fact that each representation is selectively sensitive to only part of the distortion space. The Constellation Diagram is informative about symbol clustering, phase noise spreading, and amplitude nonlinearity, but is insensitive to carrier frequency offset, which manifests as a global rotation of the entire constellation without altering its shape. The FFT Power Spectrum directly reveals frequency shifts and occupied bandwidth, but provides much less geometric information about symbol-level structure.

This motivates a multi-view formulation: instead of committing to a single representation, we align complementary views of the same signal in a shared embedding space. CLIP (Radford et al., 2021) demonstrated that contrastive alignment across very different modalities can produce strong and transferable representations. In our setting, the two views are deterministic projections of the same physical waveform, which makes the alignment problem more principled and exact than the image-text case.

We ask: **can contrastive alignment between Constellation and FFT representations produce AMC features that are simultaneously robust to distortions affecting either domain independently?**

Our contributions are:

1. We propose **DDC-AMC**, the first CLIP-style cross-domain contrastive framework for AMC that aligns Constellation Diagram and FFT Power Spectrum encoders in a shared embedding space, without any external pretrained encoder.
2. We formalize same-signal $(C_i, F_i)$ pairs as exact positive pairs for contrastive learning, grounded in the physical relationship between the IQ and frequency domains.
3. We introduce a dual-loss objective that jointly optimizes cross-domain alignment and supervised modulation classification in a single training loop.
4. We design a distortion-specific evaluation protocol and show that the dual-domain model recovers performance in conditions where each single-domain baseline individually degrades.

---

## 2. Related Work

### 2.1 Automatic Modulation Classification

The release of the RadioML dataset (O'Shea & West, 2016) catalyzed deep learning research on AMC. Early CNN-based methods on raw IQ samples demonstrated that learned features could match or exceed hand-designed pipelines. Subsequent work explored residual networks (Liu et al., 2017), CLDNN-style architectures combining convolutional, LSTM, and dense layers (Rajendran et al., 2018), and more recently Transformer-based models. Despite these improvements, most methods still operate on a single signal representation and do not explicitly leverage the complementary information present in other domains.

### 2.2 Multi-Representation Signal Classification

Related ideas have appeared in audio classification, where joint modeling of MFCCs and spectrograms has improved robustness, and in radar signal processing, where range-Doppler and micro-Doppler features are fused for waveform identification. In AMC specifically, some studies concatenate IQ-derived features or combine multiple views through late fusion. However, these approaches treat the views as independent feature sources and do not enforce any geometric alignment between their latent spaces during training.

### 2.3 Contrastive and Multi-Modal Representation Learning

Contrastive learning methods — including MoCo (He et al., 2020), SimCLR (Chen et al., 2020), and BYOL (Grill et al., 2020) — established that strong visual representations can be learned by training a model to be invariant to augmentations of the same image. CLIP (Radford et al., 2021) extended this paradigm across modalities, aligning image and text embeddings with a symmetric cross-entropy contrastive loss over paired samples. Our work applies the same alignment principle to communication signals, where the paired views are physically meaningful and mathematically related transformations of the same received waveform, making the contrastive task cleaner and more tractable than the original image-text setting.

---

## 3. Method

### 3.1 Signal Representation Extraction

Let $x(t) = I(t) + jQ(t)$ denote a received complex baseband signal consisting of $N$ samples. For each signal, we derive two fixed representations.

**Constellation Diagram.**
The complex IQ samples are rendered as a 2D histogram in the IQ plane:

$$\mathbf{C} = \operatorname{hist2d}\!\left(\{(I_n,\, Q_n)\}_{n=1}^{N}\right) \in \mathbb{R}^{H \times W}$$

The histogram is normalized to unit sum and treated as a single-channel image of size $H \times W$. This representation captures symbol clustering, phase noise spreading, and amplitude distortion as spatial patterns in IQ space.

**FFT Power Spectrum.**
We apply a Hanning window $w[n]$ to the signal before computing the discrete Fourier transform:

$$X[k] = \sum_{n=0}^{N-1} x[n]\cdot w[n]\cdot e^{-j2\pi kn/N}, \quad k = 0,\ldots,N-1$$

The one-sided log-scale power spectral density is:

$$\mathbf{F}[k] = 10\log_{10}\!\left(|X[k]|^2\right) \in \mathbb{R}^{N/2}$$

This representation captures spectral occupancy, carrier frequency offset, and harmonic structure as a 1D spectral profile.

### 3.2 Dual-Domain Contrastive Framework

Figure 1 illustrates the overall architecture of DDC-AMC. The model takes two derived representations of the same received signal — the Constellation Diagram $\mathbf{C}$ and the FFT Power Spectrum $\mathbf{F}$ — as separate inputs. These are processed by domain-specific encoders: $f_C$, a 2D CNN (ResNet-18) for the Constellation Diagram, and $f_F$, a 1D CNN for the FFT Spectrum. Each encoder produces a feature vector, which is then mapped by a projection head into a shared $d$-dimensional embedding space.

![Figure 1: Overview of the DDC-AMC Framework.](figure\Figure_1.png)

*Figure 1: Overview of the DDC-AMC framework. The Constellation Diagram $\mathbf{C}$ and FFT Power Spectrum $\mathbf{F}$ are encoded by domain-specific encoders and projected into a shared embedding space. A CLIP-style contrastive loss aligns same-signal pairs, and a classification head predicts the modulation class from the averaged fused embedding.*

Let $g_C(\cdot)$ and $g_F(\cdot)$ denote the projection heads for each domain. Their outputs are $\ell_2$-normalized embeddings $\mathbf{z}_C \in \mathbb{R}^d$ and $\mathbf{z}_F \in \mathbb{R}^d$. The model is trained jointly with two objectives: (i) a CLIP-style symmetric contrastive loss that aligns same-signal pairs $(C_i, F_i)$ and pushes apart mismatched pairs within the batch; and (ii) a supervised classification loss applied to the fused embedding

$$\mathbf{z}_{\text{fused}} = \frac{\mathbf{z}_C + \mathbf{z}_F}{2}$$

passed through a classification head $h(\cdot)$. This design keeps the representation alignment objective and the downstream classification objective tightly coupled throughout training.

### 3.3 Positive Pair Construction

For a batch of $B$ signals $\{x_i\}_{i=1}^{B}$, we extract the corresponding pairs $\{(\mathbf{C}_i, \mathbf{F}_i)\}_{i=1}^{B}$. We define:

- **Positive pair:** $(\mathbf{C}_i, \mathbf{F}_i)$ — the Constellation and FFT representations of the same signal $x_i$.
- **Negative pairs:** $(\mathbf{C}_i, \mathbf{F}_j)$ for all $j \neq i$ within the batch.

This pairing is exact and requires no augmentation policy or separate mining strategy: the positive correspondence is fully determined by the identity of the source signal.

### 3.4 Contrastive Objective

Let

$$\mathbf{z}_{C_i} = g_C(f_C(\mathbf{C}_i)), \qquad \mathbf{z}_{F_i} = g_F(f_F(\mathbf{F}_i))$$

denote the $\ell_2$-normalized projected embeddings. We define the pairwise cosine similarity matrix $S \in \mathbb{R}^{B \times B}$ as:

$$S_{ij} = \frac{\mathbf{z}_{C_i}^\top \mathbf{z}_{F_j}}{\tau}$$

where $\tau$ is a learnable temperature parameter initialized to $0.07$. The symmetric contrastive loss is:

$$\mathcal{L}_{\text{contra}} = -\frac{1}{2B} \sum_{i=1}^{B}
\left[
\log \frac{e^{S_{ii}}}{\sum_{j=1}^{B} e^{S_{ij}}}
+
\log \frac{e^{S_{ii}}}{\sum_{j=1}^{B} e^{S_{ji}}}
\right]$$

The first term trains each Constellation embedding to retrieve its paired FFT embedding; the second term does the reverse.

### 3.5 Classification Objective

The fused embedding is passed to a two-layer MLP classification head:

$$\hat{y} = h(\mathbf{z}_{\text{fused}}) \in \mathbb{R}^{M}$$

where $M = 24$ is the number of modulation classes. The classification loss is standard cross-entropy:

$$\mathcal{L}_{\text{cls}} = -\sum_{m=1}^{M} y_m \log \hat{y}_m$$

### 3.6 Total Training Objective

The two losses are combined with a scalar weight $\lambda$:

$$\mathcal{L}_{\text{DDC}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{contra}}$$

Gradients flow through both encoders and both projection heads from both terms simultaneously. The contrastive loss acts as a geometric regularizer that aligns the two representation spaces, while the classification loss drives task-specific discrimination.

### 3.7 Implementation

We summarize the full DDC-AMC pipeline in pseudocode. The procedure covers signal preprocessing, the forward pass through both encoders, loss computation, and the parameter update step for a single training iteration.

```
// ─────────────────────────────────────────────────────────────
// DDC-AMC  |  Single Training Step Pseudocode
// ─────────────────────────────────────────────────────────────
// Notation
//   B          : batch size
//   x          : raw IQ signal,  shape (B, N)       [complex]
//   C          : Constellation Diagram, shape (B, 1, H, W)
//   F_spec     : FFT Power Spectrum,    shape (B, N/2)
//   h_C, h_F   : encoder feature vectors,  shape (B, embed_dim)
//   z_C, z_F   : ℓ₂-normalized projected embeddings, shape (B, d)
//   z_fused    : averaged fused embedding,  shape (B, d)
//   S          : cosine similarity matrix,  shape (B, B)
//   τ          : learnable temperature scalar
//   λ          : contrastive loss weight scalar
// ─────────────────────────────────────────────────────────────

PROCEDURE DDC_AMC_TrainingStep(batch_x, batch_y, model, optimizer, λ):

    // ── 1. Signal Preprocessing ──────────────────────────────
    FOR EACH signal x IN batch_x:
        C      ← hist2d(Re(x), Im(x))          // 2D IQ histogram → (1, H, W) image
        C      ← normalize(C, mode=unit_sum)    // normalize to unit sum
        F_spec ← 10 · log10(|FFT(x · Hanning)|²)  // log-scale one-sided PSD → (N/2,)

    // ── 2. Encoder Forward Pass ───────────────────────────────
    h_C ← Encoder_C(C)            // 2D CNN (ResNet-18),  output: (B, embed_dim)
    h_F ← Encoder_F(F_spec)       // 1D CNN + GlobalAvgPool, output: (B, embed_dim)

    // ── 3. Projection & L2 Normalization ─────────────────────
    z_C ← L2_normalize( MLP_C(h_C) )    // Projection head g_C → (B, d)
    z_F ← L2_normalize( MLP_F(h_F) )    // Projection head g_F → (B, d)

    // ── 4. Fused Embedding & Classification ──────────────────
    z_fused ← (z_C + z_F) / 2           // element-wise average → (B, d)
    logits  ← Classifier(z_fused)        // 2-layer MLP → (B, M)

    L_cls ← CrossEntropy(logits, batch_y)

    // ── 5. CLIP-style Contrastive Loss ────────────────────────
    // Build pairwise cosine similarity matrix
    FOR i = 1 TO B:
        FOR j = 1 TO B:
            S[i, j] ← dot(z_C[i], z_F[j]) / τ    // (i == j) → positive pair

    // Symmetric cross-entropy over rows and columns
    labels ← [0, 1, 2, ..., B−1]                  // diagonal = positive pair index

    L_C→F ← CrossEntropy(S,   labels)             // retrieve F given C
    L_F→C ← CrossEntropy(S.T, labels)             // retrieve C given F

    L_contra ← (L_C→F + L_F→C) / 2

    // ── 6. Total Loss & Parameter Update ─────────────────────
    L_total ← L_cls + λ · L_contra

    optimizer.zero_grad()
    L_total.backward()          // gradients flow through both encoders
    optimizer.step()            // update Encoder_C, Encoder_F, MLP_C, MLP_F,
                                //         Classifier, τ  (all jointly)

    RETURN L_total, L_cls, L_contra

// ─────────────────────────────────────────────────────────────
// DDC-AMC  |  Inference Step Pseudocode
// ─────────────────────────────────────────────────────────────

PROCEDURE DDC_AMC_Inference(x, model):

    C      ← hist2d(Re(x), Im(x))
    C      ← normalize(C, mode=unit_sum)
    F_spec ← 10 · log10(|FFT(x · Hanning)|²)

    h_C ← Encoder_C(C)
    h_F ← Encoder_F(F_spec)

    z_C ← L2_normalize( MLP_C(h_C) )
    z_F ← L2_normalize( MLP_F(h_F) )

    z_fused  ← (z_C + z_F) / 2
    ŷ        ← argmax( Classifier(z_fused) )     // predicted modulation class

    RETURN ŷ
```

---

## 4. Experiments

### 4.1 Dataset

We evaluate on **RadioML 2018.01A**, a standard benchmark for deep learning-based AMC.

- **Modulation classes (24):** OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK
- **SNR range:** −20 dB to +30 dB in 2 dB increments
- **Samples:** 4,096 per class per SNR level; each sample contains 1,024 complex IQ values
- **Split:** 80% training, 10% validation, 10% test, stratified by class and SNR level

### 4.2 Baselines

| Model | Input Domain | Architecture |
|---|---|---|
| CNN-IQ | Raw IQ | CNN (O'Shea et al., 2016) |
| ResNet-IQ | Raw IQ | ResNet-18 |
| CLDNN | Raw IQ | CNN + LSTM + Dense |
| CNN-Const | Constellation only | ResNet-18 (2D) |
| CNN-FFT | FFT only | 1D CNN |
| **DDC-AMC (Ours)** | **Constellation + FFT** | **Dual encoder + contrastive** |

### 4.3 Evaluation Protocol

- **Overall accuracy:** average classification accuracy over all SNR levels and all 24 classes
- **Low-SNR accuracy:** accuracy restricted to SNR ≤ 0 dB
- **Distortion-specific accuracy:** accuracy evaluated separately under (i) phase noise only, (ii) frequency offset only, and (iii) concurrent phase noise and frequency offset
- **Confusion matrix:** per-class confusion at representative SNR levels (0 dB, +10 dB, +20 dB)

### 4.4 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-4}$ |
| Weight decay | $1 \times 10^{-4}$ |
| Batch size | 256 |
| Epochs | 100 |
| LR schedule | Cosine annealing |
| Temperature $\tau$ | Learnable, init $0.07$ |
| Contrastive weight $\lambda$ | $0.5$ |
| Embedding dimension | $512$ |
| Projection dimension | $128$ |
| Constellation image size | $64 \times 64$ |
| FFT input length | $512$ (one-sided) |

### 4.5 Ablation Study

To isolate the contribution of each component, we evaluate the following variants:

| Variant | Description |
|---|---|
| DDC-AMC (full) | Constellation + FFT + $\mathcal{L}_{\text{contra}}$ + $\mathcal{L}_{\text{cls}}$ |
| w/o $\mathcal{L}_{\text{contra}}$ | Dual encoder, classification loss only (late fusion baseline) |
| w/o FFT | Constellation encoder only + $\mathcal{L}_{\text{cls}}$ |
| w/o Const | FFT encoder only + $\mathcal{L}_{\text{cls}}$ |
| DDC-AMC (concat) | Replace averaged fusion with concatenation before classifier |

---

## 5. Analysis

### 5.1 Complementary Distortion Sensitivity

The two representations respond differently to common channel impairments. This is the physical motivation for their joint alignment.

| Distortion | Constellation | FFT Spectrum |
|---|---|---|
| Phase noise | Symbol cluster spreading | Spectral broadening |
| Frequency offset | Global rotation (shape unchanged) | Spectral shift |
| IQ imbalance | Asymmetric cluster geometry | Mild spectral asymmetry |
| AWGN | Increased cluster radius | Elevated noise floor |
| Nonlinear distortion | Amplitude compression | Harmonic generation |

A classifier trained on a single view inherits the blind spots of that view. For example, a model trained only on Constellation representations is insensitive to frequency offset because offset rotates the constellation without changing its inter-symbol geometry. The contrastive objective encourages both encoders to agree on a shared embedding for the same signal, so the fused representation is implicitly regularized against domain-specific failure modes.

### 5.2 Why CLIP-Style Alignment Fits AMC

In the original CLIP setting, image–text pairs are semantically related but not identical: the caption partially describes the image, and the alignment problem is inherently approximate. In DDC-AMC, the two views $\mathbf{C}_i$ and $\mathbf{F}_i$ are deterministic functions of the same waveform $x_i$, so the positive pairing is exact. This makes the contrastive task cleaner and better defined than in the cross-modal case. Furthermore, the Fourier transform provides a direct mathematical relationship between the time-domain IQ representation and the frequency-domain spectral representation, offering a principled theoretical grounding for why their embeddings should be alignable in a shared space.

### 5.3 Effect of Batch Size

The quality of the contrastive objective depends on the number of negative pairs available per positive pair, which scales linearly with batch size $B$. With $B = 256$, each positive pair is contrasted against 255 negatives in each direction. Because low-SNR modulation classes can be highly similar — especially among high-order QAM variants — a sufficiently large batch is important for the contrastive loss to expose informative negatives. We recommend a minimum batch size of 128, with 256 as the default.

---

## 6. Discussion

### 6.1 Relation to Prior Multi-View AMC Methods

Prior multi-view AMC approaches commonly combine representations through late-stage feature concatenation or ensemble voting. These methods fuse information after encoding, but do not enforce any geometric relationship between the two latent spaces during training. DDC-AMC differs structurally: the contrastive objective is applied during training to explicitly align the two embedding spaces, so that both encoders converge toward a shared representation of signal identity rather than independently optimizing for classification.

### 6.2 Limitations

**Inference cost.** DDC-AMC requires two encoder forward passes at inference time, roughly doubling computation relative to a single-encoder baseline. For latency-critical deployments on embedded SDR hardware, this may be a constraint. A natural mitigation is asymmetric inference: after training, only one encoder is used at test time, exploiting the fact that either embedding alone should carry well-aligned representations due to the contrastive objective.

**Batch size dependence.** The quality of the contrastive loss degrades with small batch sizes, as fewer negatives are available per positive pair. In memory-constrained training environments, a momentum-queue approach (MoCo-style) could decouple the number of effective negatives from the hardware batch size.

**Fixed input representations.** Both the Constellation Diagram and FFT Power Spectrum are predefined, hand-crafted transformations of the raw IQ signal. A more general future direction would jointly learn the signal-to-representation mapping alongside the contrastive alignment and classification objectives.

### 6.3 Broader Applicability

The dual-domain contrastive framework is not specific to AMC. Any signal classification task in which the received signal admits multiple physically meaningful representations with complementary distortion sensitivity is a natural candidate for this approach. Relevant applications include radar waveform classification (range-Doppler vs. micro-Doppler), industrial machinery fault detection (time-domain waveform vs. vibration spectrum), and biomedical signal analysis (raw ECG vs. frequency-domain heart rate variability features).

---

## 7. Conclusion

We presented DDC-AMC, a dual-domain contrastive learning framework for Automatic Modulation Classification. The model encodes the Constellation Diagram and FFT Power Spectrum of the same received signal using domain-specific encoders, aligns their projected embeddings with a CLIP-style symmetric contrastive loss, and predicts the modulation class from the element-wise average of the two normalized embeddings. The framework is self-contained, requires no external pretrained encoder, and is grounded in the physical complementarity of the two signal domains. We believe that principled multi-domain representation alignment is an underexplored direction in communication signal intelligence, and that DDC-AMC provides a clean and extensible foundation for this line of research.

---

## References

[1] O'Shea, T. J., & West, N. (2016). Radio machine learning dataset generation with GNU radio. *Proceedings of the GNU Radio Conference*.

[2] West, N., & O'Shea, T. (2017). Deep architectures for modulation recognition. *IEEE DySPAN 2017*.

[3] Rajendran, S., Meert, W., Giustiniano, D., Lenders, V., & Pollin, S. (2018). Deep learning models for wireless signal classification with distributed low-cost spectrum sensors. *IEEE Transactions on Cognitive Communications and Networking, 4*(3), 433–445.

[4] Liu, X., Yang, D., & El Gamal, A. (2017). Deep neural network architectures for modulation classification. *Asilomar Conference on Signals, Systems, and Computers*.

[5] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *ICML 2021*. (CLIP)

[6] He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. *CVPR 2020*. (MoCo)

[7] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML 2020*. (SimCLR)

[8] Grill, J. B., Strub, F., Altché, F., et al. (2020). Bootstrap your own latent: A new approach to self-supervised learning. *NeurIPS 2020*. (BYOL)

[9] Caron, M., Touvron, H., Misra, I., et al. (2021). Emerging properties in self-supervised vision transformers. *ICCV 2021*. (DINO)

[10] DeepSig Inc. (2018). RadioML 2018.01A Dataset. https://www.deepsig.ai/datasets

[11] Schmidl, T. M., & Cox, D. C. (1997). Robust frequency and timing synchronization for OFDM. *IEEE Transactions on Communications, 45*(12), 1613–1621.
