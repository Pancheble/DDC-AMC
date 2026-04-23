# DDC-AMC: Dual-Domain Contrastive Learning for Automatic Modulation Classification

**[English](README.md) | [Korean](README.ko.md)**

---

## Abstract

Automatic Modulation Classification (AMC) is a core problem in cognitive radio and spectrum monitoring, where the objective is to identify the modulation scheme of a received signal without prior knowledge of the transmitter. Most deep learning approaches rely on a single signal representation — such as raw IQ samples, a Constellation Diagram, or an FFT Spectrum — and therefore fail to fully exploit the complementary information available across domains.

We propose **Dual-Domain Contrastive AMC (DDC-AMC)**, a framework that encodes two physically grounded representations of the same received signal — the IQ point set and the FFT Power Spectrum — and aligns them in a shared embedding space using a CLIP-style symmetric contrastive objective. Because both views are deterministically derived from the same waveform, they form exact positive pairs for contrastive learning. The IQ point set encoder, a Set Transformer, captures phase-amplitude geometry directly from raw IQ coordinates without any rendering or binning; the FFT encoder, a 1D CNN, captures spectral energy distribution in the frequency domain. These representations respond differently to channel distortions, making their alignment useful for robust AMC. A classification head then predicts the modulation class from the element-wise average of the two $\ell_2$-normalized projected embeddings.

On the RadioML 2018.01A benchmark, DDC-AMC improves over single-domain baselines, with larger gains at low SNR and under mixed distortion conditions such as concurrent phase noise and frequency offset.

---

## 1. Introduction

Software-Defined Radio (SDR) and Cognitive Radio systems require the ability to identify modulation formats in real time, often without any prior agreement between transmitter and receiver. This task, known as Automatic Modulation Classification, sits at the intersection of signal processing and machine learning. Following the release of the RadioML dataset family by O'Shea et al., deep learning methods have largely replaced hand-crafted feature pipelines.

The standard formulation trains a network on a single representation:

$$\hat{m} = f_\theta(x), \quad x \in \{\text{IQ},\ \text{Constellation},\ \text{FFT}\}$$

where $\hat{m}$ is the predicted modulation class. While effective under favorable channel conditions, this approach is limited by the fact that each representation is selectively sensitive to only part of the distortion space. The Constellation Diagram is informative about symbol clustering, phase noise spreading, and amplitude nonlinearity, but is insensitive to carrier frequency offset, which manifests as a global rotation of the entire constellation without altering its shape. The FFT Power Spectrum directly reveals frequency shifts and occupied bandwidth, but provides much less geometric information about symbol-level structure.

This motivates a multi-view formulation: instead of committing to a single representation, we align complementary views of the same signal in a shared embedding space. CLIP (Radford et al., 2021) demonstrated that contrastive alignment across very different modalities can produce strong and transferable representations. In our setting, the two views are deterministic projections of the same physical waveform, which makes the alignment problem more principled and exact than the image-text case.

We ask: **can contrastive alignment between an IQ point set encoder and an FFT encoder produce AMC features that are simultaneously robust to distortions affecting either domain independently?**

Our contributions are:

1. We propose **DDC-AMC**, the first CLIP-style cross-domain contrastive framework for AMC that aligns a permutation-invariant IQ point set encoder and an FFT Power Spectrum encoder in a shared embedding space, without any external pretrained encoder.
2. We formalize same-signal $(C_i, F_i)$ pairs — where $C_i$ is a raw IQ point set and $F_i$ is its corresponding FFT spectrum — as exact positive pairs for contrastive learning, grounded in the physical relationship between the IQ and frequency domains.
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

**IQ Point Set (Constellation Domain).**
The complex IQ samples are treated directly as an unordered set of two-dimensional coordinates:

$$\mathbf{C} = \{(I_n,\, Q_n)\}_{n=1}^{N} \in \mathbb{R}^{N \times 2}$$

No rendering or binning is applied. This representation preserves the exact phase-amplitude geometry of each received symbol, capturing symbol clustering, phase noise spreading, and amplitude distortion without quantization loss. Because the identity of a constellation is fully determined by the set of occupied IQ positions — not by the order in which symbols arrive — the representation is treated as a permutation-invariant point set.

**FFT Power Spectrum.**
We apply a Hann window $w[n]$ to the signal before computing the discrete Fourier transform:

$$X[k] = \sum_{n=0}^{N-1} x[n]\cdot w[n]\cdot e^{-j2\pi kn/N}, \quad k = 0,\ldots,N-1$$

The one-sided log-scale power spectral density is:

$$\mathbf{F}[k] = 10\log_{10}\!\left(|X[k]|^2\right) \in \mathbb{R}^{N/2}$$

This representation captures spectral occupancy, carrier frequency offset, and harmonic structure as a 1D spectral profile.

### 3.2 Dual-Domain Contrastive Framework

DDC-AMC processes the same received signal through two parallel branches, each designed to match the geometric structure of its input domain.

The first branch handles the IQ point set $\mathbf{C} \in \mathbb{R}^{N \times 2}$. Because the identity of a constellation is determined by the set of occupied IQ positions and not by the temporal order of symbol arrival, the encoder $f_C$ must be permutation-invariant. We therefore adopt a Set Transformer, which applies attention-based pooling over the $N$ input points and produces a fixed-length feature vector $\mathbf{h}_C \in \mathbb{R}^{\text{embed\_dim}}$ regardless of point ordering.

The second branch handles the FFT power spectrum $\mathbf{F} \in \mathbb{R}^{N/2}$. Unlike the IQ point set, the spectrum is an ordered sequence in which the position of each bin carries physical meaning — bin index directly encodes frequency. Permutation invariance would destroy this structure. The encoder $f_F$ is therefore a 1D CNN with global average pooling, which preserves frequency-axis locality through its convolutional receptive fields and produces a feature vector $\mathbf{h}_F \in \mathbb{R}^{\text{embed\_dim}}$.

Each feature vector is passed through an independent projection head — $g_C(\cdot)$ and $g_F(\cdot)$ respectively — implemented as two-layer MLPs mapping to a shared $d$-dimensional space. The outputs are $\ell_2$-normalized to produce unit-norm embeddings $\mathbf{z}_C \in \mathbb{R}^d$ and $\mathbf{z}_F \in \mathbb{R}^d$.

The model is then trained jointly with two objectives. First, a CLIP-style symmetric contrastive loss aligns same-signal pairs $(\mathbf{C}_i, \mathbf{F}_i)$ in the shared embedding space and pushes apart mismatched pairs within the batch. Second, the two embeddings are fused by element-wise averaging,

$$\mathbf{z}_{\text{fused}} = \frac{\mathbf{z}_C + \mathbf{z}_F}{2}$$

and passed through a classification head $h(\cdot)$ to predict the modulation class. This design keeps the representation alignment objective and the downstream classification objective tightly coupled throughout training.

### 3.3 Positive Pair Construction

For a batch of $B$ signals $\{x_i\}_{i=1}^{B}$, we extract the corresponding pairs $\{(\mathbf{C}_i, \mathbf{F}_i)\}_{i=1}^{B}$, where $\mathbf{C}_i \in \mathbb{R}^{N \times 2}$ is the IQ point set and $\mathbf{F}_i \in \mathbb{R}^{N/2}$ is the FFT power spectrum of signal $x_i$. We define:

- **Positive pair:** $(\mathbf{C}_i, \mathbf{F}_i)$ — the IQ point set and FFT representations of the same signal $x_i$.
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

The first term trains each IQ point set embedding to retrieve its paired FFT embedding; the second term does the reverse.

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
//   x          : raw IQ signal,              shape (B, N)    [complex]
//   C          : IQ point set,               shape (B, N, 2)
//   F_spec     : FFT Power Spectrum,         shape (B, N/2)
//   h_C, h_F   : encoder feature vectors,   shape (B, embed_dim)
//   z_C, z_F   : ℓ₂-normalized embeddings, shape (B, d)
//   z_fused    : averaged fused embedding,  shape (B, d)
//   S          : cosine similarity matrix,  shape (B, B)
//   τ          : learnable temperature scalar
//   λ          : contrastive loss weight scalar
// ─────────────────────────────────────────────────────────────

PROCEDURE DDC_AMC_TrainingStep(batch_x, batch_y, model, optimizer, λ):

    // ── 1. Signal Preprocessing ──────────────────────────────
    FOR EACH signal x IN batch_x:
        C      ← stack(Re(x), Im(x), axis=-1)       // IQ coordinates → (N, 2)
        F_spec ← 10 · log10(|FFT(x · Hanning)|²)    // log-scale one-sided PSD → (N/2,)

    // ── 2. Encoder Forward Pass ───────────────────────────────
    h_C ← Encoder_C(C)            // Set Transformer (permutation-invariant)
                                   //   input: (B, N, 2) → output: (B, embed_dim)
    h_F ← Encoder_F(F_spec)       // 1D CNN + GlobalAvgPool
                                   //   input: (B, N/2)  → output: (B, embed_dim)

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

    C      ← stack(Re(x), Im(x), axis=-1)           // (N, 2)
    F_spec ← 10 · log10(|FFT(x · Hanning)|²)        // (N/2,)

    h_C ← Encoder_C(C)                              // Set Transformer
    h_F ← Encoder_F(F_spec)                         // 1D CNN

    z_C ← L2_normalize( MLP_C(h_C) )
    z_F ← L2_normalize( MLP_F(h_F) )

    z_fused  ← (z_C + z_F) / 2
    ŷ        ← argmax( Classifier(z_fused) )         // predicted modulation class

    RETURN ŷ
```

---

## References

[1] O'Shea, T. J., & West, N. (2016). Radio Machine Learning Dataset Generation with GNU Radio. *Proceedings of the 6th GNU Radio Conference*, Vol. 1, No. 1.

[2] West, N. E., & O'Shea, T. J. (2017). Deep Architectures for Modulation Recognition. *Proceedings of the 2017 IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN)*.

[3] Rajendran, S., Meert, W., Giustiniano, D., Lenders, V., & Pollin, S. (2018). Deep Learning Models for Wireless Signal Classification with Distributed Low-Cost Spectrum Sensors. *IEEE Transactions on Cognitive Communications and Networking, 4*(3), 433–445.

[4] Liu, X., Yang, D., & El Gamal, A. (2017). Deep Neural Network Architectures for Modulation Classification. *Proceedings of the 51st Asilomar Conference on Signals, Systems, and Computers (ACSSC)*.

[5] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *arXiv preprint arXiv:2103.00020*.

[6] He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 9726–9735.

[7] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *Proceedings of the 37th International Conference on Machine Learning (ICML)*.

[8] Grill, J.-B., Strub, F., Altché, F., et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. *Advances in Neural Information Processing Systems (NeurIPS)*.

[9] Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97, 3744–3753.

[10] DeepSig. RadioML 2018.01A Dataset. Available from the official DeepSig datasets page.

[11] Schmidl, T. M., & Cox, D. C. (1997). Robust Frequency and Timing Synchronization for OFDM. *IEEE Transactions on Communications, 45*(12), 1613–1621.
