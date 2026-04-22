# DDC-AMC: Dual-Domain Contrastive Learning for Automatic Modulation Classification

**[English](README.md) | [Korean](README.ko.md)**

---

## 초록

자동 변조 분류(Automatic Modulation Classification, AMC)는 인지 무선(Cognitive Radio) 및 스펙트럼 모니터링 분야의 핵심 문제로, 송신기에 대한 사전 정보 없이 수신 신호의 변조 방식을 식별하는 것을 목표로 한다. 기존의 딥러닝 기반 접근법은 원시 IQ 샘플, 컨스텔레이션 다이어그램(Constellation Diagram), 또는 FFT 스펙트럼 중 단일 신호 표현에 의존하는 경향이 있으며, 이로 인해 각 도메인에 내재된 상호보완적 정보를 충분히 활용하지 못하는 한계를 지닌다.

본 논문에서는 **이중 도메인 대조 학습 기반 AMC(Dual-Domain Contrastive AMC, DDC-AMC)** 프레임워크를 제안한다. 제안 방법은 동일한 수신 신호로부터 물리적으로 유의미한 두 가지 표현, 즉 컨스텔레이션 다이어그램과 FFT 전력 스펙트럼을 각각 인코딩하고, CLIP 방식의 대칭적 대조 목적 함수를 통해 공유 임베딩 공간에서 정렬한다. 두 표현 모두 동일한 파형으로부터 결정론적으로 유도되므로, 대조 학습에 있어 정확한 양성 쌍(positive pair)을 구성한다. 컨스텔레이션 인코더는 IQ 공간에서의 위상-진폭 클러스터링을 포착하는 반면, FFT 인코더는 주파수 도메인에서의 스펙트럼 에너지 분포를 포착한다. 두 표현은 채널 왜곡에 상이하게 반응하므로, 이들의 정렬은 강인한 AMC를 위해 유효하다. 이후 분류 헤드(classification head)는 두 $\ell_2$-정규화된 투영 임베딩의 원소별 평균으로부터 변조 방식을 예측한다.

RadioML 2018.01A 벤치마크 실험에서, DDC-AMC는 단일 도메인 기준 모델 대비 향상된 분류 성능을 달성하였으며, 특히 저 SNR 환경 및 위상 잡음과 주파수 오프셋이 동시에 존재하는 복합 왜곡 조건에서 더욱 뚜렷한 성능 향상이 관찰되었다.

---

## 1. 서론

소프트웨어 정의 라디오(Software-Defined Radio, SDR) 및 인지 무선 시스템은 송수신기 간 사전 협약 없이 수신 신호의 변조 방식을 실시간으로 식별하는 능력을 요구한다. 자동 변조 분류로 알려진 이 과제는 신호 처리와 기계 학습의 교차점에 위치하며, O'Shea 등의 RadioML 데이터셋 공개를 계기로 딥러닝 기반 방법론이 수공 특징 추출(hand-crafted feature) 파이프라인을 대체하는 흐름이 가속화되었다.

표준적인 AMC 정식화는 단일 표현에 대해 신경망을 학습한다:

$$\hat{m} = f_\theta(x), \quad x \in \{\text{IQ},\ \text{Constellation},\ \text{FFT}\}$$

여기서 $\hat{m}$은 예측된 변조 방식을 나타낸다. 이 접근법은 양호한 채널 조건에서는 효과적이나, 각 표현이 왜곡 공간의 일부에만 선택적으로 민감하다는 구조적 한계를 지닌다. 컨스텔레이션 다이어그램은 심볼 클러스터링, 위상 잡음 확산, 진폭 비선형성에 대한 정보를 풍부하게 담고 있는 반면, 컨스텔레이션 전체가 회전하더라도 그 형태가 변하지 않는 반송파 주파수 오프셋에는 둔감하다. FFT 전력 스펙트럼은 주파수 편이와 점유 대역폭을 직접적으로 드러내지만, 심볼 수준의 기하학적 구조에 관한 정보는 상대적으로 부족하다.

이러한 관찰은 다중 뷰 정식화(multi-view formulation)의 필요성을 제기한다. 단일 표현에 국한되는 대신, 동일 신호의 상호보완적 뷰를 공유 임베딩 공간에서 정렬하는 접근법을 취한다. CLIP(Radford et al., 2021)은 매우 상이한 모달리티 간 대조 정렬이 강력하고 전이 가능한 표현을 생성함을 입증하였다. 본 연구의 경우, 두 뷰는 동일한 물리적 파형의 결정론적 투영이므로, 이미지-텍스트 쌍의 경우보다 정렬 문제가 더욱 정교하고 명확하게 정의된다.

본 논문이 제기하는 핵심 질문은 다음과 같다: **컨스텔레이션과 FFT 표현 간의 대조 정렬이, 각 도메인에 개별적으로 영향을 미치는 왜곡에 대해 동시에 강인한 AMC 특징을 생성할 수 있는가?**

본 논문의 기여는 다음과 같이 요약된다:

1. 외부 사전 학습 인코더 없이 컨스텔레이션 다이어그램 인코더와 FFT 전력 스펙트럼 인코더를 공유 임베딩 공간에 정렬하는, AMC를 위한 최초의 CLIP 방식 교차 도메인 대조 프레임워크인 **DDC-AMC**를 제안한다.
2. 동일 신호 쌍 $(C_i, F_i)$를 대조 학습의 정확한 양성 쌍으로 정형화하고, 이를 IQ 도메인과 주파수 도메인 간의 물리적 관계에 근거하여 정당화한다.
3. 교차 도메인 정렬과 지도 변조 분류를 단일 학습 루프 내에서 공동으로 최적화하는 이중 손실 목적 함수를 도입한다.
4. 왜곡 유형별 평가 프로토콜을 설계하고, 각 단일 도메인 기준 모델이 개별적으로 성능이 저하되는 조건에서 이중 도메인 모델이 성능을 회복함을 보인다.

---

## 2. 관련 연구

### 2.1 자동 변조 분류

RadioML 데이터셋(O'Shea & West, 2016)의 공개는 AMC에 대한 딥러닝 연구를 촉진하였다. 원시 IQ 샘플에 대한 초기 CNN 기반 방법들은 학습된 특징이 수공 설계된 파이프라인에 필적하거나 이를 능가할 수 있음을 입증하였다. 이후 연구들은 잔차 신경망(Liu et al., 2017), 합성곱, LSTM, 완전연결층을 결합한 CLDNN 구조(Rajendran et al., 2018), 그리고 최근의 트랜스포머 기반 모델을 탐구하였다. 이러한 발전에도 불구하고, 대부분의 방법은 여전히 단일 신호 표현에 의존하며 다른 도메인에 존재하는 상호보완적 정보를 명시적으로 활용하지 않는다.

### 2.2 다중 표현 신호 분류

유사한 아이디어는 MFCC와 스펙트로그램의 결합 모델링이 강인성을 향상시킨 오디오 분류, 그리고 파형 식별을 위해 거리-도플러(range-Doppler) 및 마이크로-도플러(micro-Doppler) 특징을 결합하는 레이더 신호 처리 분야에서도 나타난다. AMC 분야에서도 일부 연구는 IQ 파생 특징을 연결하거나 후기 융합(late fusion)을 통해 다중 뷰를 결합하는 방식을 제안하였다. 그러나 이러한 접근법들은 두 뷰를 독립적인 특징 원천으로 취급하며, 학습 과정에서 두 잠재 공간 간의 기하학적 정렬을 명시적으로 강제하지 않는다.

### 2.3 대조 학습 및 다중 모달 표현 학습

MoCo(He et al., 2020), SimCLR(Chen et al., 2020), BYOL(Grill et al., 2020) 등의 대조 학습 방법들은 동일한 이미지의 증강 변환에 불변하도록 모델을 학습함으로써 강력한 시각 표현을 학습할 수 있음을 확립하였다. CLIP(Radford et al., 2021)은 이 패러다임을 모달리티 간으로 확장하여, 쌍으로 이루어진 샘플에 대한 대칭적 교차 엔트로피 대조 손실로 이미지와 텍스트 임베딩을 정렬하였다. 본 연구는 동일한 정렬 원리를 통신 신호에 적용하며, 여기서 쌍을 이루는 뷰는 동일한 수신 파형의 물리적으로 유의미하고 수학적으로 관련된 변환으로서, 원래의 이미지-텍스트 설정보다 대조 과제를 더욱 명확하고 다루기 용이하게 만든다.

---

## 3. 제안 방법

### 3.1 신호 표현 추출

$N$개의 샘플로 구성된 수신 복소 기저대역 신호를 $x(t) = I(t) + jQ(t)$로 표기한다. 각 신호로부터 두 가지 고정된 표현을 유도한다.

**컨스텔레이션 다이어그램.**
복소 IQ 샘플을 IQ 평면에서 2차원 히스토그램으로 렌더링한다:

$$\mathbf{C} = \operatorname{hist2d}\!\left(\{(I_n,\, Q_n)\}_{n=1}^{N}\right) \in \mathbb{R}^{H \times W}$$

히스토그램은 합이 1이 되도록 정규화되며, $H \times W$ 크기의 단채널 이미지로 취급된다. 이 표현은 IQ 공간에서의 공간적 패턴으로서 심볼 클러스터링, 위상 잡음 확산, 진폭 왜곡을 포착한다.

**FFT 전력 스펙트럼.**
이산 푸리에 변환을 계산하기에 앞서 신호에 Hanning 윈도우 $w[n]$을 적용한다:

$$X[k] = \sum_{n=0}^{N-1} x[n]\cdot w[n]\cdot e^{-j2\pi kn/N}, \quad k = 0,\ldots,N-1$$

단측(one-sided) 로그 스케일 전력 스펙트럼 밀도는 다음과 같이 정의된다:

$$\mathbf{F}[k] = 10\log_{10}\!\left(|X[k]|^2\right) \in \mathbb{R}^{N/2}$$

이 표현은 1차원 스펙트럼 프로파일로서 스펙트럼 점유, 반송파 주파수 오프셋, 고조파 구조를 포착한다.

### 3.2 이중 도메인 대조 프레임워크

그림 1은 DDC-AMC의 전체 구조를 도시한다. 모델은 동일한 수신 신호로부터 유도된 두 표현, 즉 컨스텔레이션 다이어그램 $\mathbf{C}$와 FFT 전력 스펙트럼 $\mathbf{F}$를 별도의 입력으로 받는다. 이들은 도메인별 인코더, 즉 컨스텔레이션 다이어그램을 위한 2D CNN($f_C$, ResNet-18)과 FFT 스펙트럼을 위한 1D CNN($f_F$)에 의해 처리된다. 각 인코더는 특징 벡터를 출력하며, 이는 투영 헤드를 통해 공유 $d$차원 임베딩 공간으로 매핑된다.

![그림 1: DDC-AMC 프레임워크 개요.](figure\Figure_1.png)

*그림 1: DDC-AMC 프레임워크 개요. 컨스텔레이션 다이어그램 $\mathbf{C}$와 FFT 전력 스펙트럼 $\mathbf{F}$는 도메인별 인코더에 의해 인코딩되고 공유 임베딩 공간에 투영된다. CLIP 방식의 대조 손실이 동일 신호 쌍을 정렬하며, 분류 헤드는 평균 융합 임베딩으로부터 변조 방식을 예측한다.*

각 도메인의 투영 헤드를 $g_C(\cdot)$와 $g_F(\cdot)$로 표기한다. 이들의 출력은 $\ell_2$-정규화된 임베딩 $\mathbf{z}_C \in \mathbb{R}^d$와 $\mathbf{z}_F \in \mathbb{R}^d$이다. 모델은 두 가지 목적 함수로 공동 학습된다: (i) 동일 신호 쌍 $(C_i, F_i)$를 정렬하고 배치 내 불일치 쌍을 분리하는 CLIP 방식의 대칭 대조 손실, 그리고 (ii) 융합 임베딩

$$\mathbf{z}_{\text{fused}} = \frac{\mathbf{z}_C + \mathbf{z}_F}{2}$$

에 분류 헤드 $h(\cdot)$를 적용하는 지도 분류 손실. 이 설계는 표현 정렬 목적 함수와 하위 분류 목적 함수가 학습 전반에 걸쳐 긴밀하게 결합되도록 한다.

### 3.3 양성 쌍 구성

$B$개의 신호 배치 $\{x_i\}_{i=1}^{B}$에 대해, 대응하는 쌍 $\{(\mathbf{C}_i, \mathbf{F}_i)\}_{i=1}^{B}$를 추출한다. 다음과 같이 정의한다:

- **양성 쌍:** $(\mathbf{C}_i, \mathbf{F}_i)$ — 동일한 신호 $x_i$의 컨스텔레이션 및 FFT 표현.
- **음성 쌍:** 배치 내 모든 $j \neq i$에 대한 $(\mathbf{C}_i, \mathbf{F}_j)$.

이 쌍 구성은 정확하며, 별도의 데이터 증강 정책이나 마이닝 전략을 요구하지 않는다. 양성 대응 관계는 원본 신호의 동일성에 의해 완전히 결정된다.

### 3.4 대조 목적 함수

$\ell_2$-정규화된 투영 임베딩을 다음과 같이 표기한다:

$$\mathbf{z}_{C_i} = g_C(f_C(\mathbf{C}_i)), \qquad \mathbf{z}_{F_i} = g_F(f_F(\mathbf{F}_i))$$

쌍별 코사인 유사도 행렬 $S \in \mathbb{R}^{B \times B}$를 다음과 같이 정의한다:

$$S_{ij} = \frac{\mathbf{z}_{C_i}^\top \mathbf{z}_{F_j}}{\tau}$$

여기서 $\tau$는 $0.07$로 초기화되는 학습 가능한 온도 파라미터이다. 대칭적 대조 손실은 다음과 같다:

$$\mathcal{L}_{\text{contra}} = -\frac{1}{2B} \sum_{i=1}^{B}
\left[
\log \frac{e^{S_{ii}}}{\sum_{j=1}^{B} e^{S_{ij}}}
+
\log \frac{e^{S_{ii}}}{\sum_{j=1}^{B} e^{S_{ji}}}
\right]$$

첫 번째 항은 각 컨스텔레이션 임베딩이 대응하는 FFT 임베딩을 검색하도록 학습하며, 두 번째 항은 그 역방향을 수행한다.

### 3.5 분류 목적 함수

융합 임베딩은 2층 MLP 분류 헤드에 입력된다:

$$\hat{y} = h(\mathbf{z}_{\text{fused}}) \in \mathbb{R}^{M}$$

여기서 $M = 24$는 변조 방식의 클래스 수이다. 분류 손실은 표준 교차 엔트로피를 사용한다:

$$\mathcal{L}_{\text{cls}} = -\sum_{m=1}^{M} y_m \log \hat{y}_m$$

### 3.6 전체 학습 목적 함수

두 손실은 스칼라 가중치 $\lambda$로 결합된다:

$$\mathcal{L}_{\text{DDC}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{contra}}$$

기울기는 두 항으로부터 두 인코더와 두 투영 헤드 모두를 통해 동시에 역전파된다. 대조 손실은 두 표현 공간을 정렬하는 기하학적 정규화 항으로 작용하며, 분류 손실은 과제 특화 판별성을 유도한다.

### 3.7 구현

DDC-AMC의 전체 파이프라인을 의사 코드(pseudocode)로 요약한다. 아래 절차는 신호 전처리, 두 인코더를 통한 순전파, 손실 계산, 그리고 단일 학습 반복에서의 파라미터 갱신 단계를 포함한다.

```
// ─────────────────────────────────────────────────────────────
// DDC-AMC  |  단일 학습 스텝 의사 코드
// ─────────────────────────────────────────────────────────────
// 표기
//   B          : 배치 크기
//   x          : 원시 IQ 신호,         shape (B, N)       [복소수]
//   C          : 컨스텔레이션 다이어그램, shape (B, 1, H, W)
//   F_spec     : FFT 전력 스펙트럼,      shape (B, N/2)
//   h_C, h_F   : 인코더 특징 벡터,      shape (B, embed_dim)
//   z_C, z_F   : ℓ₂-정규화 투영 임베딩, shape (B, d)
//   z_fused    : 평균 융합 임베딩,       shape (B, d)
//   S          : 코사인 유사도 행렬,     shape (B, B)
//   τ          : 학습 가능한 온도 스칼라
//   λ          : 대조 손실 가중치 스칼라
// ─────────────────────────────────────────────────────────────

PROCEDURE DDC_AMC_TrainingStep(batch_x, batch_y, model, optimizer, λ):

    // ── 1. 신호 전처리 ────────────────────────────────────────
    FOR EACH signal x IN batch_x:
        C      ← hist2d(Re(x), Im(x))               // 2D IQ 히스토그램 → (1, H, W) 이미지
        C      ← normalize(C, mode=unit_sum)         // 합이 1이 되도록 정규화
        F_spec ← 10 · log10(|FFT(x · Hanning)|²)    // 로그 스케일 단측 PSD → (N/2,)

    // ── 2. 인코더 순전파 ──────────────────────────────────────
    h_C ← Encoder_C(C)            // 2D CNN (ResNet-18),       출력: (B, embed_dim)
    h_F ← Encoder_F(F_spec)       // 1D CNN + GlobalAvgPool,   출력: (B, embed_dim)

    // ── 3. 투영 및 L2 정규화 ─────────────────────────────────
    z_C ← L2_normalize( MLP_C(h_C) )    // 투영 헤드 g_C → (B, d)
    z_F ← L2_normalize( MLP_F(h_F) )    // 투영 헤드 g_F → (B, d)

    // ── 4. 융합 임베딩 및 분류 ───────────────────────────────
    z_fused ← (z_C + z_F) / 2           // 원소별 평균 → (B, d)
    logits  ← Classifier(z_fused)        // 2층 MLP → (B, M)

    L_cls ← CrossEntropy(logits, batch_y)

    // ── 5. CLIP 방식 대조 손실 ───────────────────────────────
    // 쌍별 코사인 유사도 행렬 구성
    FOR i = 1 TO B:
        FOR j = 1 TO B:
            S[i, j] ← dot(z_C[i], z_F[j]) / τ    // (i == j) → 양성 쌍

    // 행 및 열에 대한 대칭 교차 엔트로피
    labels ← [0, 1, 2, ..., B−1]                  // 대각선 = 양성 쌍 인덱스

    L_C→F ← CrossEntropy(S,   labels)             // C가 주어졌을 때 F 검색
    L_F→C ← CrossEntropy(S.T, labels)             // F가 주어졌을 때 C 검색

    L_contra ← (L_C→F + L_F→C) / 2

    // ── 6. 전체 손실 및 파라미터 갱신 ───────────────────────
    L_total ← L_cls + λ · L_contra

    optimizer.zero_grad()
    L_total.backward()          // 기울기가 두 인코더 모두를 통해 역전파됨
    optimizer.step()            // Encoder_C, Encoder_F, MLP_C, MLP_F,
                                //   Classifier, τ 공동 갱신

    RETURN L_total, L_cls, L_contra

// ─────────────────────────────────────────────────────────────
// DDC-AMC  |  추론 스텝 의사 코드
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
    ŷ        ← argmax( Classifier(z_fused) )     // 예측 변조 방식

    RETURN ŷ
```

---

## 4. 실험

### 4.1 데이터셋

**RadioML 2018.01A**를 평가 데이터셋으로 사용한다. 이는 딥러닝 기반 AMC 연구의 표준 벤치마크이다.

- **변조 방식(24종):** OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK
- **SNR 범위:** −20 dB에서 +30 dB까지 2 dB 간격
- **샘플 수:** 클래스 및 SNR 수준별 각 4,096개; 각 샘플은 1,024개의 복소 IQ 값으로 구성
- **분할:** 클래스 및 SNR 수준에 따라 층화 추출된 학습 80%, 검증 10%, 테스트 10%

### 4.2 기준 모델

| 모델 | 입력 도메인 | 구조 |
|---|---|---|
| CNN-IQ | 원시 IQ | CNN (O'Shea et al., 2016) |
| ResNet-IQ | 원시 IQ | ResNet-18 |
| CLDNN | 원시 IQ | CNN + LSTM + 완전연결층 |
| CNN-Const | 컨스텔레이션만 | ResNet-18 (2D) |
| CNN-FFT | FFT만 | 1D CNN |
| **DDC-AMC (제안)** | **컨스텔레이션 + FFT** | **이중 인코더 + 대조 학습** |

### 4.3 평가 프로토콜

- **전체 정확도:** 모든 SNR 수준 및 24개 클래스에 대한 평균 분류 정확도
- **저 SNR 정확도:** SNR ≤ 0 dB로 제한된 분류 정확도
- **왜곡 유형별 정확도:** (i) 위상 잡음만 존재하는 경우, (ii) 주파수 오프셋만 존재하는 경우, (iii) 위상 잡음과 주파수 오프셋이 동시에 존재하는 경우 각각 평가
- **혼동 행렬:** 대표적 SNR 수준(0 dB, +10 dB, +20 dB)에서의 클래스별 혼동 분석

### 4.4 학습 설정

| 하이퍼파라미터 | 값 |
|---|---|
| 최적화 알고리즘 | AdamW |
| 학습률 | $1 \times 10^{-4}$ |
| 가중치 감쇠 | $1 \times 10^{-4}$ |
| 배치 크기 | 256 |
| 에폭 수 | 100 |
| 학습률 스케줄 | 코사인 어닐링 |
| 온도 $\tau$ | 학습 가능, 초기값 $0.07$ |
| 대조 손실 가중치 $\lambda$ | $0.5$ |
| 임베딩 차원 | $512$ |
| 투영 차원 | $128$ |
| 컨스텔레이션 이미지 크기 | $64 \times 64$ |
| FFT 입력 길이 | $512$ (단측) |

### 4.5 소거 연구

각 구성 요소의 기여도를 분리하기 위해 다음 변형 모델들을 평가한다:

| 변형 모델 | 설명 |
|---|---|
| DDC-AMC (전체) | 컨스텔레이션 + FFT + $\mathcal{L}_{\text{contra}}$ + $\mathcal{L}_{\text{cls}}$ |
| w/o $\mathcal{L}_{\text{contra}}$ | 분류 손실만 사용하는 이중 인코더 (후기 융합 기준 모델) |
| w/o FFT | 컨스텔레이션 인코더만 + $\mathcal{L}_{\text{cls}}$ |
| w/o Const | FFT 인코더만 + $\mathcal{L}_{\text{cls}}$ |
| DDC-AMC (concat) | 분류기 이전의 평균 융합을 연결(concatenation)로 대체 |

---

## 5. 분석

### 5.1 상호보완적 왜곡 민감도

두 표현은 일반적인 채널 손상에 상이하게 반응한다. 이는 두 표현의 공동 정렬을 위한 물리적 동기를 제공한다.

| 왜곡 유형 | 컨스텔레이션 | FFT 스펙트럼 |
|---|---|---|
| 위상 잡음 | 심볼 클러스터 확산 | 스펙트럼 광대화 |
| 주파수 오프셋 | 전체 회전 (형태 불변) | 스펙트럼 편이 |
| IQ 불균형 | 비대칭 클러스터 기하 | 경미한 스펙트럼 비대칭 |
| AWGN | 클러스터 반경 증가 | 잡음 기저 상승 |
| 비선형 왜곡 | 진폭 압축 | 고조파 생성 |

단일 뷰로 학습된 분류기는 해당 뷰의 맹점을 그대로 상속한다. 예를 들어, 컨스텔레이션 표현만으로 학습된 모델은 주파수 오프셋에 둔감한데, 이는 오프셋이 심볼 간 기하 구조를 변형하지 않은 채 컨스텔레이션 전체를 회전시키기 때문이다. 대조 목적 함수는 두 인코더가 동일한 신호에 대해 공유 임베딩에 합의하도록 장려하므로, 융합 표현은 도메인 특화 실패 양상에 대해 암묵적으로 정규화된다.

### 5.2 CLIP 방식 정렬이 AMC에 적합한 이유

원래 CLIP 설정에서 이미지-텍스트 쌍은 의미론적으로 관련되어 있으나 동일하지는 않다. 설명문은 이미지를 부분적으로 기술하며, 정렬 문제는 본질적으로 근사적이다. DDC-AMC에서 두 뷰 $\mathbf{C}_i$와 $\mathbf{F}_i$는 동일한 파형 $x_i$의 결정론적 함수이므로, 양성 쌍 구성이 정확하다. 이는 대조 과제를 교차 모달 경우보다 더욱 명확하고 잘 정의된 문제로 만든다. 더불어 푸리에 변환은 시간 도메인 IQ 표현과 주파수 도메인 스펙트럼 표현 사이의 직접적인 수학적 관계를 제공하므로, 두 임베딩이 공유 공간에서 정렬될 수 있는 이론적 근거를 원리적으로 뒷받침한다.

### 5.3 배치 크기의 영향

대조 목적 함수의 품질은 양성 쌍당 이용 가능한 음성 쌍의 수에 의존하며, 이는 배치 크기 $B$에 선형 비례한다. $B = 256$일 때, 각 양성 쌍은 각 방향으로 255개의 음성 쌍과 대비된다. 저 SNR 환경에서 변조 방식 클래스들은 매우 유사할 수 있으며, 특히 고차 QAM 변형들 사이에서 그러하다. 따라서 대조 손실이 유익한 음성 쌍을 노출하기 위해서는 충분히 큰 배치가 중요하다. 최소 배치 크기로 128을 권장하며, 기본값으로는 256을 사용한다.

---

## 6. 논의

### 6.1 기존 다중 뷰 AMC 방법과의 관계

기존의 다중 뷰 AMC 접근법들은 일반적으로 후기 단계의 특징 연결(feature concatenation)이나 앙상블 투표를 통해 표현을 결합한다. 이러한 방법들은 인코딩 이후에 정보를 융합하지만, 학습 과정에서 두 잠재 공간 사이의 기하학적 관계를 명시적으로 강제하지 않는다. DDC-AMC는 구조적으로 다르다. 대조 목적 함수는 학습 중에 두 임베딩 공간을 명시적으로 정렬하기 위해 적용되므로, 두 인코더가 분류를 독립적으로 최적화하는 대신 신호 정체성의 공유 표현을 향해 수렴하게 된다.

### 6.2 한계점

**추론 비용.** DDC-AMC는 추론 시 두 번의 인코더 순전파를 요구하므로, 단일 인코더 기준 모델 대비 연산량이 약 두 배로 증가한다. 지연 시간이 엄격히 제한되는 임베디드 SDR 하드웨어 배포 환경에서는 이것이 제약 요인이 될 수 있다. 자연스러운 완화 방법은 비대칭 추론으로, 학습 완료 후 테스트 시에는 하나의 인코더만 사용하는 것이다. 이는 대조 목적 함수로 인해 어느 임베딩 단독으로도 잘 정렬된 표현을 보유할 것이라는 사실을 활용한다.

**배치 크기 의존성.** 대조 손실의 품질은 소규모 배치에서 저하되는데, 이는 양성 쌍당 이용 가능한 음성 쌍의 수가 감소하기 때문이다. 메모리 제약이 있는 학습 환경에서는 모멘텀 큐 방식(MoCo 스타일)을 통해 유효 음성 쌍의 수를 하드웨어 배치 크기와 분리할 수 있다.

**고정된 입력 표현.** 컨스텔레이션 다이어그램과 FFT 전력 스펙트럼 모두 원시 IQ 신호의 사전 정의된 수공 변환이다. 보다 일반적인 향후 방향으로는 대조 정렬 및 분류 목적 함수와 함께 신호-표현 매핑을 공동으로 학습하는 접근법을 고려할 수 있다.

### 6.3 광범위한 적용 가능성

이중 도메인 대조 프레임워크는 AMC에 특정되지 않는다. 수신 신호가 상호보완적 왜곡 민감도를 갖는 다수의 물리적으로 유의미한 표현을 허용하는 모든 신호 분류 과제가 이 접근법의 자연스러운 적용 후보이다. 관련 응용 분야로는 레이더 파형 분류(거리-도플러 대 마이크로-도플러), 산업 기계 결함 탐지(시간 도메인 파형 대 진동 스펙트럼), 생체 의료 신호 분석(원시 ECG 대 주파수 도메인 심박 변이도 특징) 등이 있다.

---

## 7. 결론

본 논문에서는 자동 변조 분류를 위한 이중 도메인 대조 학습 프레임워크인 DDC-AMC를 제시하였다. 제안 모델은 도메인별 인코더를 이용하여 동일한 수신 신호의 컨스텔레이션 다이어그램과 FFT 전력 스펙트럼을 각각 인코딩하고, CLIP 방식의 대칭 대조 손실로 투영 임베딩을 정렬하며, 두 정규화 임베딩의 원소별 평균으로부터 변조 방식을 예측한다. 본 프레임워크는 외부 사전 학습 인코더를 필요로 하지 않는 자기완결적 구조이며, 두 신호 도메인의 물리적 상보성에 근거한다. 원리적인 다중 도메인 표현 정렬이 통신 신호 지능 분야에서 충분히 탐구되지 않은 방향임을 확신하며, DDC-AMC가 이 연구 흐름을 위한 명확하고 확장 가능한 기반을 제공하기를 기대한다.

---

## 참고 문헌

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
