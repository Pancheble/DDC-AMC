# DDC-AMC: 이중 도메인 대조 학습 기반 자동 변조 분류

**[English](README.md) | [Korean](README.ko.md)**

---

## 초록

자동 변조 분류(Automatic Modulation Classification, AMC)는 인지 무선 및 스펙트럼 모니터링에서 핵심적인 문제로, 수신 신호의 변조 방식을 송신기 정보 없이 식별하는 것을 목표로 한다. 기존의 심층학습 방법은 원시 IQ 샘플, 성상도(Constellation Diagram), 또는 FFT 스펙트럼과 같은 단일 표현에 의존하는 경우가 많아, 서로 보완적인 정보를 충분히 활용하지 못한다.

본 연구는 동일한 수신 신호로부터 결정적으로 도출되는 두 가지 물리적으로 의미 있는 표현, 즉 IQ 점 집합과 FFT 전력 스펙트럼을 인코딩하고, CLIP 스타일의 대칭 대조 목적함수를 이용하여 공유 임베딩 공간에 정렬하는 **Dual-Domain Contrastive AMC (DDC-AMC)** 를 제안한다. 두 관점은 동일한 파형에서 정확히 유도되므로, 대조 학습에 있어서 완전한 양성 쌍을 이룬다. IQ 점 집합 인코더인 Set Transformer는 별도의 렌더링이나 비닝 없이 원시 IQ 좌표의 위상-진폭 기하를 직접 포착하며, FFT 인코더인 1D CNN은 주파수 영역의 스펙트럼 에너지 분포를 포착한다. 이 두 표현은 채널 왜곡에 서로 다르게 반응하므로, 이를 정렬하는 과정은 강인한 AMC에 유리하다. 이후 분류 헤드는 두 $\ell_2$-정규화 투영 임베딩의 원소별 평균으로부터 변조 클래스를 예측한다.

RadioML 2018.01A 벤치마크에서 DDC-AMC는 단일 도메인 기준선 대비 성능을 향상시키며, 특히 낮은 SNR 조건과 위상 잡음 및 주파수 오프셋이 동시에 존재하는 혼합 왜곡 조건에서 더 큰 이득을 보인다.

---

## 1. 서론

소프트웨어 정의 무선(Software-Defined Radio, SDR) 및 인지 무선(Cognitive Radio) 시스템은 사전에 송수신기 간 합의가 없는 상황에서도 변조 형식을 실시간으로 식별할 수 있어야 한다. 자동 변조 분류는 이러한 요구를 충족하기 위한 문제로서, 신호 처리와 기계학습의 교차점에 위치한다. O'Shea et al.에 의해 RadioML 데이터셋 계열이 공개된 이후, 심층학습 방법이 수작업 특징 추출 파이프라인을 점차 대체해 왔다.

표준적인 정식화는 단일 표현만을 입력으로 받는 네트워크를 학습한다.

$$\hat{m} = f_\theta(x), \quad x \in \{\text{IQ},\ \text{Constellation},\ \text{FFT}\}$$

여기서 $\hat{m}$은 예측된 변조 클래스이다. 이러한 접근은 특정 채널 조건 하에서는 효과적이지만, 각 표현이 왜곡 공간의 일부에만 선택적으로 민감하다는 한계를 가진다. 성상도는 심볼 군집, 위상 잡음에 의한 확산, 진폭 비선형성에는 유용하지만, 반송파 주파수 오프셋은 성상도의 전체 회전으로만 나타나기 때문에 그 형태 자체는 크게 변하지 않는다. 반면 FFT 전력 스펙트럼은 주파수 이동과 점유 대역폭을 직접 드러내지만, 심볼 수준의 기하학적 구조에 대한 정보는 훨씬 적다.

이에 따라 본 연구는 단일 표현에 고정되지 않고, 동일 신호의 보완적 관점을 공유 임베딩 공간에서 정렬하는 다중 뷰 정식화를 채택한다. CLIP(Radford et al., 2021)은 서로 매우 다른 모달리티 간에도 대조 정렬을 통해 강력하고 전이 가능한 표현을 학습할 수 있음을 보였다. 본 연구의 경우 두 관점은 동일한 물리적 파형으로부터 결정적으로 생성되므로, 이미지-텍스트 쌍보다 훨씬 더 원리적이고 정확한 정렬 문제를 제공한다.

본 연구는 다음과 같은 질문에서 출발한다. **IQ 점 집합 인코더와 FFT 인코더 사이의 대조 정렬이, 두 도메인 중 어느 한쪽에만 영향을 주는 왜곡에 대해서도 동시에 강인한 AMC 특징을 형성할 수 있는가?**

본 연구의 기여는 다음과 같다.

1. 본 연구는 외부 사전학습 인코더 없이, IQ 점 집합 인코더와 FFT 전력 스펙트럼 인코더를 공유 임베딩 공간에 정렬하는, AMC를 위한 최초의 CLIP 스타일 교차 도메인 대조 학습 프레임워크인 **DDC-AMC**를 제안한다.
2. 동일 신호에서 얻어진 $(C_i, F_i)$ 쌍, 즉 $C_i$는 원시 IQ 점 집합이고 $F_i$는 이에 대응하는 FFT 스펙트럼이라는 점을 활용하여, 이를 대조 학습을 위한 정확한 양성 쌍으로 정식화한다. 이 정식화는 IQ 및 주파수 도메인 사이의 물리적 관계에 기반한다.
3. 교차 도메인 정렬과 지도 변조 분류를 단일 학습 루프에서 동시에 최적화하는 이중 손실 목적함수를 제안한다.
4. 왜곡 특화 평가 프로토콜을 설계하고, 각 단일 도메인 기준선이 개별적으로 성능 저하를 보이는 조건에서 이중 도메인 모델이 성능을 회복함을 보인다.

---

## 2. 관련 연구

### 2.1 자동 변조 분류

RadioML 데이터셋의 공개는 AMC에 대한 심층학습 연구를 촉진하였다. 초기의 원시 IQ 샘플 기반 CNN 방법들은 학습된 특징이 수작업 파이프라인과 동등하거나 그 이상일 수 있음을 보여 주었다. 이후 연구는 residual network(Liu et al., 2017), convolutional, LSTM, dense layer를 결합한 CLDNN 계열 구조(Rajendran et al., 2018), 그리고 최근의 Transformer 기반 모델로 확장되었다. 그러나 이러한 방법들 역시 대부분 단일 신호 표현에 의존하며, 다른 도메인에 존재하는 보완 정보를 명시적으로 활용하지 않는다.

### 2.2 다중 표현 기반 신호 분류

유사한 아이디어는 오디오 분류에서도 관찰되었는데, MFCC와 spectrogram의 결합 모델이 강인성을 개선한 바 있다. 레이더 신호 처리에서는 range-Doppler와 micro-Doppler 특징을 융합하여 파형을 식별하는 연구가 있다. AMC에서도 일부 연구는 IQ 기반 특징을 연결하거나 여러 관점을 late fusion으로 결합한다. 다만 이러한 접근은 관점을 독립적인 특징 소스로 취급할 뿐, 학습 과정에서 잠재 공간의 기하학적 정렬을 강제하지는 않는다.

### 2.3 대조 학습 및 멀티모달 표현 학습

MoCo(He et al., 2020), SimCLR(Chen et al., 2020), BYOL(Grill et al., 2020)을 포함한 대조 학습 방법들은 동일한 이미지의 증강 관점을 구분하도록 학습함으로써 강력한 시각 표현을 학습할 수 있음을 보였다. CLIP(Radford et al., 2021)은 이 패러다임을 모달리티 간으로 확장하여, 짝지어진 샘플에 대해 대칭적 교차 엔트로피 대조 손실을 사용해 이미지와 텍스트 임베딩을 정렬하였다. 본 연구는 이러한 정렬 원리를 통신 신호에 적용한다. 여기서 짝지어진 관점은 동일 수신 파형의 물리적으로 의미 있는, 그리고 수학적으로 관련된 변환이므로, 원래의 이미지-텍스트 설정보다 대조 과제가 더 명확하고 다루기 쉽다.

---

## 3. 방법

### 3.1 신호 표현 추출

수신 복소 기저대역 신호를 $x(t) = I(t) + jQ(t)$라 하자. 신호마다 다음의 두 가지 고정 표현을 도출한다.

**IQ 점 집합(성상도 영역).**
복소 IQ 샘플은 순서가 없는 2차원 좌표 집합으로 직접 다룬다.

$$\mathbf{C} = \{(I_n,\, Q_n)\}_{n=1}^{N} \in \mathbb{R}^{N \times 2}$$

여기서는 별도의 렌더링이나 비닝을 수행하지 않는다. 이 표현은 수신된 각 심볼의 정확한 위상-진폭 기하를 보존하며, 양자화 손실 없이 심볼 군집, 위상 잡음 확산, 진폭 왜곡을 포착한다. 성상도의 정체성은 심볼이 도착한 순서가 아니라, 점유된 IQ 위치의 집합에 의해 결정되므로, 이 표현은 순열에 대해 불변인 point set으로 취급된다.

**FFT 전력 스펙트럼.**
신호에 해닝(Hann) 창 $w[n]$을 적용한 뒤 이산 푸리에 변환을 계산한다.

$$X[k] = \sum_{n=0}^{N-1} x[n]\cdot w[n]\cdot e^{-j2\pi kn/N}, \quad k = 0,\ldots,N-1$$

한쪽 스펙트럼의 로그 스케일 전력 스펙트럼 밀도는 다음과 같다.

$$\mathbf{F}[k] = 10\log_{10}\!\left(|X[k]|^2\right) \in \mathbb{R}^{N/2}$$

이 표현은 스펙트럼 점유, 반송파 주파수 오프셋, 그리고 고조파 구조를 1차원 스펙트럼 프로파일로 포착한다.

### 3.2 이중 도메인 대조 학습 프레임워크

DDC-AMC는 동일한 수신 신호를 두 개의 병렬 브랜치에 통과시키며, 각 브랜치는 입력 도메인의 기하 구조에 맞게 설계된다.

첫 번째 브랜치는 IQ 점 집합 $\mathbf{C} \in \mathbb{R}^{N \times 2}$를 처리한다. 성상도의 정체성은 심볼 도착 순서가 아니라 점유된 IQ 위치 집합에 의해 결정되므로, 인코더 $f_C$는 순열 불변이어야 한다. 이에 따라 본 연구는 attention 기반 pooling을 적용하는 Set Transformer를 채택하여, 입력 점의 순서와 무관하게 고정 길이 특징 벡터 $\mathbf{h}_C \in \mathbb{R}^{\text{embed\_dim}}$를 산출한다.

두 번째 브랜치는 FFT 전력 스펙트럼 $\mathbf{F} \in \mathbb{R}^{N/2}$를 처리한다. IQ 점 집합과 달리 스펙트럼은 각 bin의 위치가 물리적 의미를 가지는 순서화된 시퀀스이다. 따라서 순열 불변성을 적용하면 이러한 구조가 소실된다. 이에 따라 인코더 $f_F$는 전역 평균 풀링을 포함한 1D CNN으로 구성하며, 합성곱 수용영역을 통해 주파수축의 국소성을 보존한 채 특징 벡터 $\mathbf{h}_F \in \mathbb{R}^{\text{embed\_dim}}$를 생성한다.

각 특징 벡터는 별도의 projection head, 즉 $g_C(\cdot)$와 $g_F(\cdot)$를 거치며, 이는 공유 $d$차원 공간으로 사상하는 2층 MLP로 구현된다. 출력은 $\ell_2$ 정규화되어 단위 노름 임베딩 $\mathbf{z}_C \in \mathbb{R}^d$와 $\mathbf{z}_F \in \mathbb{R}^d$를 형성한다.

이후 모델은 두 목적함수를 결합하여 공동 학습된다. 첫째, CLIP 스타일의 대칭 대조 손실은 동일 신호 쌍 $(\mathbf{C}_i, \mathbf{F}_i)$를 공유 임베딩 공간에 정렬하고, 배치 내 비정합 쌍은 멀어지도록 학습한다. 둘째, 두 임베딩을 원소별 평균으로 융합한다.

$$\mathbf{z}_{\text{fused}} = \frac{\mathbf{z}_C + \mathbf{z}_F}{2}$$

그 후 분류 헤드 $h(\cdot)$에 입력하여 변조 클래스를 예측한다. 이 설계는 표현 정렬 목표와 다운스트림 분류 목표를 학습 전 과정에서 긴밀하게 결합한다.

### 3.3 양성 쌍 구성

배치 크기 $B$에 대해 신호 집합 $\{x_i\}_{i=1}^{B}$로부터 대응하는 쌍 $\{(\mathbf{C}_i, \mathbf{F}_i)\}_{i=1}^{B}$를 추출한다. 여기서 $\mathbf{C}_i \in \mathbb{R}^{N \times 2}$는 IQ 점 집합이고, $\mathbf{F}_i \in \mathbb{R}^{N/2}$는 신호 $x_i$의 FFT 전력 스펙트럼이다. 이를 다음과 같이 정의한다.

- **양성 쌍:** $(\mathbf{C}_i, \mathbf{F}_i)$ — 동일 신호 $x_i$에서 얻은 IQ 점 집합과 FFT 표현.
- **음성 쌍:** $(\mathbf{C}_i, \mathbf{F}_j)$, 모든 $j \neq i$에 대해 배치 내의 나머지 쌍.

이 쌍 구성은 정확하며 별도의 증강 정책이나 샘플 마이닝 전략을 필요로 하지 않는다. 양성 대응은 원 신호의 동일성만으로 완전히 결정된다.

### 3.4 대조 목적함수

정규화된 투영 임베딩을 다음과 같이 두자.

$$\mathbf{z}_{C_i} = g_C(f_C(\mathbf{C}_i)), \qquad \mathbf{z}_{F_i} = g_F(f_F(\mathbf{F}_i))$$

쌍별 코사인 유사도 행렬 $S \in \mathbb{R}^{B \times B}$는 다음과 같이 정의한다.

$$S_{ij} = \frac{\mathbf{z}_{C_i}^\top \mathbf{z}_{F_j}}{\tau}$$

여기서 $\tau$는 학습 가능한 temperature 파라미터이며, 초기값은 $0.07$로 설정한다. 대칭 대조 손실은 다음과 같다.

$$\mathcal{L}_{\text{contra}} = -\frac{1}{2B} \sum_{i=1}^{B}
\left[
\log \frac{e^{S_{ii}}}{\sum_{j=1}^{B} e^{S_{ij}}}
+
\log \frac{e^{S_{ii}}}{\sum_{j=1}^{B} e^{S_{ji}}}
\right]$$

첫 번째 항은 각 IQ 점 집합 임베딩이 대응하는 FFT 임베딩을 찾도록 학습하고, 두 번째 항은 그 반대 방향을 학습한다.

### 3.5 분류 목적함수

융합 임베딩은 2층 MLP 분류 헤드에 입력된다.

$$\hat{y} = h(\mathbf{z}_{\text{fused}}) \in \mathbb{R}^{M}$$

여기서 $M = 24$는 변조 클래스의 수이다. 분류 손실은 표준 교차 엔트로피이다.

$$\mathcal{L}_{\text{cls}} = -\sum_{m=1}^{M} y_m \log \hat{y}_m$$

### 3.6 전체 학습 목표

두 손실은 스칼라 가중치 $\lambda$로 결합된다.

$$\mathcal{L}_{\text{DDC}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{contra}}$$

기울기는 두 인코더와 두 projection head를 통해 동시에 전달된다. 대조 손실은 두 표현 공간을 정렬하는 기하학적 정규화 항으로 작용하며, 분류 손실은 과제 특화적인 판별성을 유도한다.

### 3.7 구현

전체 DDC-AMC 파이프라인을 의사코드로 정리하면 다음과 같다. 이 절차는 신호 전처리, 양쪽 인코더를 통한 순전파, 손실 계산, 그리고 단일 학습 반복에 대한 파라미터 갱신 단계를 포함한다.

```
// ─────────────────────────────────────────────────────────────
// DDC-AMC  |  단일 학습 스텝 의사코드
// ─────────────────────────────────────────────────────────────
// 표기
//   B          : 배치 크기
//   x          : 원시 IQ 신호,               shape (B, N)    [복소수]
//   C          : IQ 점 집합,                 shape (B, N, 2)
//   F_spec     : FFT 전력 스펙트럼,          shape (B, N/2)
//   h_C, h_F   : 인코더 특징 벡터,           shape (B, embed_dim)
//   z_C, z_F   : ℓ₂-정규화 임베딩,          shape (B, d)
//   z_fused    : 평균 융합 임베딩,           shape (B, d)
//   S          : 코사인 유사도 행렬,         shape (B, B)
//   τ          : 학습 가능한 temperature 스칼라
//   λ          : 대조 손실 가중치 스칼라
// ─────────────────────────────────────────────────────────────

PROCEDURE DDC_AMC_TrainingStep(batch_x, batch_y, model, optimizer, λ):

    // ── 1. 신호 전처리 ─────────────────────────────────────
    FOR EACH signal x IN batch_x:
        C      ← stack(Re(x), Im(x), axis=-1)       // IQ 좌표 → (N, 2)
        F_spec ← 10 · log10(|FFT(x · Hanning)|²)    // 로그 스케일 one-sided PSD → (N/2,)

    // ── 2. 인코더 순전파 ───────────────────────────────────
    h_C ← Encoder_C(C)            // Set Transformer (순열 불변)
                                   //   입력: (B, N, 2) → 출력: (B, embed_dim)
    h_F ← Encoder_F(F_spec)       // 1D CNN + GlobalAvgPool
                                   //   입력: (B, N/2)  → 출력: (B, embed_dim)

    // ── 3. Projection 및 L2 정규화 ───────────────────────
    z_C ← L2_normalize( MLP_C(h_C) )    // projection head g_C → (B, d)
    z_F ← L2_normalize( MLP_F(h_F) )    // projection head g_F → (B, d)

    // ── 4. 융합 임베딩 및 분류 ───────────────────────────
    z_fused ← (z_C + z_F) / 2           // 원소별 평균 → (B, d)
    logits  ← Classifier(z_fused)       // 2층 MLP → (B, M)

    L_cls ← CrossEntropy(logits, batch_y)

    // ── 5. CLIP 스타일 대조 손실 ─────────────────────────
    // 쌍별 코사인 유사도 행렬 구성
    FOR i = 1 TO B:
        FOR j = 1 TO B:
            S[i, j] ← dot(z_C[i], z_F[j]) / τ    // (i == j) → 양성 쌍

    // 행/열 방향에 대한 대칭 교차 엔트로피
    labels ← [0, 1, 2, ..., B−1]                  // 대각선 = 양성 쌍 인덱스

    L_C→F ← CrossEntropy(S,   labels)             // C로부터 F를 검색
    L_F→C ← CrossEntropy(S.T, labels)             // F로부터 C를 검색

    L_contra ← (L_C→F + L_F→C) / 2

    // ── 6. 전체 손실 및 파라미터 갱신 ─────────────────────
    L_total ← L_cls + λ · L_contra

    optimizer.zero_grad()
    L_total.backward()          // 기울기는 두 인코더를 통해 역전파됨
    optimizer.step()            // Encoder_C, Encoder_F, MLP_C, MLP_F,
                                // Classifier, τ 를 공동 갱신

    RETURN L_total, L_cls, L_contra

// ─────────────────────────────────────────────────────────────
// DDC-AMC  |  추론 스텝 의사코드
// ─────────────────────────────────────────────────────────────

PROCEDURE DDC_AMC_Inference(x, model):

    C      ← stack(Re(x), Im(x), axis=-1)           // (N, 2)
    F_spec ← 10 · log10(|FFT(x · Hanning)|²)        // (N/2,)

    h_C ← Encoder_C(C)                              // Set Transformer
    h_F ← Encoder_F(F_spec)                         // 1D CNN

    z_C ← L2_normalize( MLP_C(h_C) )
    z_F ← L2_normalize( MLP_F(h_F) )

    z_fused  ← (z_C + z_F) / 2
    ŷ        ← argmax( Classifier(z_fused) )         // 예측된 변조 클래스

    RETURN ŷ
```

---

## 참고문헌

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
