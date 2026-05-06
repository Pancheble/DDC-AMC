# DDC-AMC: Dual-Domain Contrastive Learning for Automatic Modulation Classification

**[English](README.md) | [Korean](README.ko.md)**

---

## 초록

자동 변조 분류(Automatic Modulation Classification, AMC)는 인지 무선 및 스펙트럼 모니터링의 핵심 문제로, 송신기에 대한 사전 지식 없이 수신 신호의 변조 방식을 식별하는 것을 목표로 한다. 대부분의 딥러닝 기반 접근법은 원시 IQ 샘플, 성상도(Constellation Diagram), 또는 FFT 스펙트럼과 같은 단일 신호 표현에 의존하므로, 서로 상보적인 도메인 정보를 충분히 활용하지 못한다.

본 논문에서는 동일한 수신 신호로부터 물리적으로 정당화된 두 표현, 즉 IQ 포인트 집합과 FFT 전력 스펙트럼을 인코딩하고, CLIP 스타일의 대칭적 대조 목적함수를 이용하여 이를 공유 임베딩 공간에 정렬하는 **이중 도메인 대조 AMC(Dual-Domain Contrastive AMC, DDC-AMC)** 프레임워크를 제안한다. 두 관점은 동일한 파형으로부터 결정적으로 도출되므로, 대조 학습에 있어 정확한 양성 쌍을 이룬다. IQ 포인트 집합 인코더로는 Set Transformer를 사용하여 렌더링이나 binning 없이 원시 IQ 좌표로부터 위상-진폭 기하를 직접 포착하도록 하였고, FFT 인코더로는 1D CNN을 사용하여 주파수 영역의 스펙트럼 에너지 분포를 포착하도록 하였다. 두 표현은 채널 왜곡에 대해 서로 다른 방식으로 반응하므로, 이들의 정렬은 보다 강건한 AMC를 목표로 한다. 이후 분류 헤드는 두 개의 $\ell_2$ 정규화 투영 임베딩의 원소별 평균으로부터 변조 클래스를 예측한다.

본 프레임워크는 RadioML 2018.01A 벤치마크를 대상으로 평가하는 것을 목표로 하며, 특히 낮은 SNR 조건과 위상 잡음 및 주파수 편이의 동시 발생과 같은 복합 왜곡 조건에 초점을 둔다.

---

## 1. 서론

소프트웨어 정의 무선(Software-Defined Radio, SDR) 및 인지 무선(Cognitive Radio) 시스템은 송수신기 간의 사전 합의가 없는 상황에서도 실시간으로 변조 형식을 식별할 수 있는 능력을 요구한다. 이러한 과제는 자동 변조 분류(Automatic Modulation Classification)라 하며, 신호 처리와 기계 학습의 교차점에 위치한다. O'Shea 등과 함께 RadioML 데이터셋 계열이 공개된 이후, 수작업 특징 공학보다 딥러닝 방법이 널리 사용되기 시작하였다.

표준적인 문제 설정은 단일 표현에 대해 네트워크를 학습하는 방식이다.

$$\hat{m} = f_\theta(x), \quad x \in \{\text{IQ},\ \text{Constellation},\ \text{FFT}\}$$

여기서 $\hat{m}$은 예측된 변조 클래스이다. 이 접근법은 유리한 채널 조건에서는 효과적이지만, 각 표현이 왜곡 공간의 일부에만 선택적으로 민감하다는 한계를 지닌다. 성상도는 심볼 군집, 위상 잡음 확산, 진폭 비선형성에 대한 정보를 제공하지만, 캐리어 주파수 편이에는 둔감하다. 이는 성상도의 전체 형상을 바꾸지 않은 채 전역 회전으로만 나타난다. 반면 FFT 전력 스펙트럼은 주파수 이동과 점유 대역폭을 직접적으로 드러내지만, 심볼 수준의 기하학적 구조에 대한 정보는 훨씬 적다.

이러한 한계는 다중 관점(multi-view) 정식을 동기화한다. 즉, 하나의 표현에 고정되지 않고 동일한 신호의 상보적 관점을 공유 임베딩 공간에 정렬하는 것이다. CLIP(Radford et al., 2021)은 서로 매우 다른 모달리티 간의 대조적 정렬이 강력하고 전이 가능한 표현을 생성할 수 있음을 보였다. 본 연구의 경우 두 관점은 동일한 물리적 파형으로부터 결정적으로 생성되는 변환이므로, 이미지-텍스트 사례보다 더 원칙적이고 정확한 정렬 문제로 볼 수 있다.

따라서 우리는 다음과 같은 질문을 제기한다. **IQ 포인트 집합 인코더와 FFT 인코더 사이의 대조적 정렬을 통해, 각 도메인에 독립적으로 영향을 주는 왜곡에 대해 안정적인 AMC 특징을 구성할 수 있는가?**

본 연구의 기여는 다음과 같이 정리된다.

1. 본 논문은 외부 사전학습 인코더에 의존하지 않고, 순열 불변 IQ 포인트 집합 인코더와 FFT 전력 스펙트럼 인코더를 공유 임베딩 공간에 정렬하는 CLIP 스타일의 교차 도메인 대조 학습 기반 AMC 프레임워크 **DDC-AMC**를 제안한다.
2. IQ 도메인과 주파수 도메인 사이의 물리적 관계를 바탕으로, 동일 신호 $(C_i, F_i)$ 쌍 — 여기서 $C_i$는 원시 IQ 포인트 집합이고 $F_i$는 이에 대응하는 FFT 스펙트럼이다 — 을 대조 학습을 위한 정확한 양성 쌍으로 정식화한다.
3. 교차 도메인 정렬과 지도 변조 분류를 단일 학습 루프에서 공동으로 최적화하는 이중 손실 목적함수를 제안한다.
4. 왜곡 특화 평가 프로토콜을 정의하고, 단일 도메인 기준선이 각각 저하되는 조건에서 이중 도메인 모델이 성능을 더 잘 유지하는지 검증하고자 한다.

---

## 2. 관련 연구

### 2.1 자동 변조 분류

RadioML 데이터셋(O'Shea & West, 2016)의 공개는 AMC 연구에 딥러닝의 확산을 촉진하였다. 원시 IQ 샘플에 대한 초기 CNN 기반 방법들은 학습된 특징이 수작업 파이프라인과 동등하거나 이를 상회할 수 있음을 보였다. 이후 연구에서는 잔차 네트워크(Liu et al., 2017), 합성곱·LSTM·완전연결 계층을 결합한 CLDNN 구조(Rajendran et al., 2018), 그리고 최근에는 Transformer 기반 모델까지 탐구되었다. 그러나 이러한 방법들 대부분은 여전히 단일 신호 표현에 의존하며, 다른 도메인에 존재하는 상보적 정보를 명시적으로 활용하지는 않는다.

### 2.2 다중 표현 신호 분류

유사한 아이디어는 오디오 분류 분야에서도 나타났으며, MFCC와 스펙트로그램의 공동 모델링이 강건성을 향상시킨 사례가 있다. 레이더 신호 처리에서는 range-Doppler 및 micro-Doppler 특징을 결합하여 파형을 식별하는 방법이 연구되었다. AMC 영역에서도 IQ 기반 특징을 연결하거나 다중 관점을 후기 결합(late fusion)하는 연구가 존재한다. 다만 이러한 접근법은 대체로 관점을 독립적인 특징 소스로 취급하며, 학습 과정에서 잠재 공간 간의 기하학적 정렬을 강제하지는 않는다.

### 2.3 대조 학습 및 다중 모달 표현 학습

MoCo(He et al., 2020), SimCLR(Chen et al., 2020), BYOL(Grill et al., 2020)과 같은 대조 학습 방법은 동일 이미지의 증강에 대해 불변성을 갖도록 모델을 학습함으로써 강력한 시각 표현을 얻을 수 있음을 보였다. CLIP(Radford et al., 2021)은 이 패러다임을 모달리티 전반으로 확장하여, 짝지어진 샘플에 대해 대칭적 교차 엔트로피 대조 손실을 적용함으로써 이미지와 텍스트 임베딩을 정렬하였다. 본 연구는 이러한 정렬 원리를 통신 신호에 적용한다. 이때 짝지어진 관점들은 동일한 수신 파형의 물리적으로 의미 있고 수학적으로 관련된 변환이므로, 원래의 이미지-텍스트 설정보다 대조 과제가 더 명료하고 다루기 용이하다.

---

## 3. 방법

### 3.1 신호 표현 추출

$N$개의 샘플로 이루어진 수신 복소 기저대역 신호를 $x(t) = I(t) + jQ(t)$라고 하자. 각 신호에 대해 우리는 두 가지 고정된 표현을 도출한다.

**IQ 포인트 집합(성상도 영역).**
복소 IQ 샘플을 두 차원 좌표의 순서 없는 집합으로 직접 취급한다.

$$\mathbf{C} = \{(I_n,\, Q_n)\}_{n=1}^{N} \in \mathbb{R}^{N \times 2}$$

여기에는 렌더링이나 binning이 적용되지 않는다. 이 표현은 각 수신 심볼의 정확한 위상-진폭 기하를 보존하며, 양자화 손실 없이 심볼 군집, 위상 잡음 확산, 진폭 왜곡을 포착한다. 성상도의 정체성은 심볼이 도착하는 순서가 아니라 점유된 IQ 위치 집합에 의해 완전히 결정되므로, 이 표현은 순열 불변(point set)으로 취급된다.

**FFT 전력 스펙트럼.**
이산 푸리에 변환을 계산하기 전에 신호에 Hann window $w[n]$을 적용한다.

$$X[k] = \sum_{n=0}^{N-1} x[n]\cdot w[n]\cdot e^{-j2\pi kn/N}, \quad k = 0,\ldots,N-1$$

단측 로그 스케일 전력 스펙트럼 밀도는 다음과 같다.

$$\mathbf{F}[k] = 10\log_{10}\!\left(|X[k]|^2\right) \in \mathbb{R}^{N/2}$$

이 표현은 스펙트럼 점유, 캐리어 주파수 편이, 조화 구조를 1차원 스펙트럼 프로파일로 포착한다.

### 3.2 이중 도메인 대조 프레임워크

DDC-AMC는 동일한 수신 신호를 두 개의 병렬 분기(parallel branch)로 처리하며, 각 분기는 입력 도메인의 기하 구조에 부합하도록 설계된다.

첫 번째 분기는 IQ 포인트 집합 $\mathbf{C} \in \mathbb{R}^{N \times 2}$를 다룬다. 성상도의 정체성은 심볼 도착 순서가 아니라 점유된 IQ 위치 집합에 의해 결정되므로, 인코더 $f_C$는 순열 불변이어야 한다. 따라서 우리는 $N$개의 입력 포인트에 대해 attention 기반 pooling을 수행하고, 포인트 순서와 무관하게 고정 길이 특징 벡터 $\mathbf{h}_C \in \mathbb{R}^{\mathrm{embed\_dim}}$를 생성하는 Set Transformer를 채택한다.

두 번째 분기는 FFT 전력 스펙트럼 $\mathbf{F} \in \mathbb{R}^{N/2}$를 다룬다. IQ 포인트 집합과 달리, 스펙트럼은 각 bin의 위치가 물리적 의미를 가지는 순차열이다. 즉, bin index는 주파수를 직접적으로 인코딩한다. 따라서 순열 불변성은 이러한 구조를 훼손할 수 있다. 이에 따라 인코더 $f_F$는 전역 평균 풀링(global average pooling)을 갖는 1D CNN으로 구성하여, 합성곱 수용 영역을 통해 주파수 축의 국소성을 보존하고 특징 벡터 $\mathbf{h}_F \in \mathbb{R}^{\mathrm{embed\_dim}}$를 생성하도록 한다.

각 특징 벡터는 별도의 projection head — 각각 $g_C(\cdot)$와 $g_F(\cdot)$ — 를 통과하며, 이는 공유된 $d$차원 공간으로 사상하는 2층 MLP로 구현된다. 이후 출력은 $\ell_2$ 정규화되어 단위 노름 임베딩 $\mathbf{z}_C \in \mathbb{R}^d$와 $\mathbf{z}_F \in \mathbb{R}^d$를 형성한다.

모델은 두 목적을 공동으로 최적화한다. 첫째, CLIP 스타일의 대칭적 대조 손실은 동일 신호 쌍 $(\mathbf{C}_i, \mathbf{F}_i)$를 공유 임베딩 공간에서 정렬하고, 배치 내 불일치 쌍을 서로 멀어지도록 한다. 둘째, 두 임베딩은 원소별 평균을 통해 융합된다.

$$\mathbf{z}_{\text{fused}} = \frac{\mathbf{z}_C + \mathbf{z}_F}{2}$$

이후 분류 헤드 $h(\cdot)$에 입력되어 변조 클래스를 예측한다. 이러한 설계는 표현 정렬 목적과 하위 분류 목적이 학습 전반에 걸쳐 긴밀하게 결합되도록 한다.

### 3.3 양성 쌍 구성

배치 크기가 $B$인 신호 집합 $\{x_i\}_{i=1}^{B}$에 대해, 우리는 대응하는 쌍 $\{(\mathbf{C}_i, \mathbf{F}_i)\}_{i=1}^{B}$를 추출한다. 여기서 $\mathbf{C}_i \in \mathbb{R}^{N \times 2}$는 IQ 포인트 집합이고, $\mathbf{F}_i \in \mathbb{R}^{N/2}$는 신호 $x_i$의 FFT 전력 스펙트럼이다. 우리는 다음과 같이 정의한다.

- **양성 쌍:** $(\mathbf{C}_i, \mathbf{F}_i)$ — 동일한 신호 $x_i$의 IQ 포인트 집합 표현과 FFT 표현.
- **음성 쌍:** $(\mathbf{C}_i, \mathbf{F}_j)$, 단 $j \neq i$인 배치 내 모든 경우.

이 쌍 구성은 정확하며, 별도의 증강 정책이나 샘플 마이닝 전략을 요구하지 않는다. 양성 대응은 원천 신호의 동일성에 의해 완전히 결정된다.

### 3.4 대조 목적함수

다음과 같이 $\ell_2$ 정규화된 투영 임베딩을 정의한다.

$$\mathbf{z}_{C_i} = g_C(f_C(\mathbf{C}_i)), \qquad \mathbf{z}_{F_i} = g_F(f_F(\mathbf{F}_i))$$

쌍별 코사인 유사도 행렬 $S \in \mathbb{R}^{B \times B}$는 다음과 같다.

$$S_{ij} = \frac{\mathbf{z}_{C_i}^\top \mathbf{z}_{F_j}}{\tau}$$

여기서 $\tau$는 0.07로 초기화되는 학습 가능한 온도 매개변수이다. 대칭적 대조 손실은 다음과 같다.

$$\mathcal{L}_{\text{contra}} = -\frac{1}{2B} \sum_{i=1}^{B}
\left[
\log \frac{e^{S_{ii}}}{\sum_{j=1}^{B} e^{S_{ij}}}
+
\log \frac{e^{S_{ii}}}{\sum_{j=1}^{B} e^{S_{ji}}}
\right]$$

첫 번째 항은 각 IQ 포인트 집합 임베딩이 대응되는 FFT 임베딩을 검색하도록 학습하며, 두 번째 항은 그 반대 방향을 학습한다.

### 3.5 분류 목적함수

융합 임베딩은 2층 MLP 분류 헤드에 입력된다.

$$\hat{y} = h(\mathbf{z}_{\text{fused}}) \in \mathbb{R}^{M}$$

여기서 $M = 24$는 변조 클래스의 수이다. 분류 손실은 표준 교차 엔트로피로 정의된다.

$$\mathcal{L}_{\text{cls}} = -\sum_{m=1}^{M} y_m \log \hat{y}_m$$

### 3.6 전체 학습 목적함수

두 손실은 스칼라 가중치 $\lambda$를 통해 결합된다.

$$\mathcal{L}_{\text{DDC}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{contra}}$$

두 손실항의 그래디언트는 두 인코더와 두 projection head 전체로 동시에 전파된다. 대조 손실은 두 표현 공간을 정렬하는 기하학적 정규화 항으로 작용하며, 분류 손실은 과제 특화 판별력을 유도한다.

### 3.7 구현

아래에서는 DDC-AMC 전체 파이프라인을 의사코드로 요약한다. 이 절차는 신호 전처리, 두 인코더를 통한 순전파, 손실 계산, 그리고 단일 학습 반복에 대한 파라미터 갱신 단계를 포함한다.

```
// ─────────────────────────────────────────────────────────────
// DDC-AMC  |  단일 학습 스텝 의사코드
// ─────────────────────────────────────────────────────────────
// 기호 설명
//   B          : 배치 크기
//   x          : 원시 IQ 신호,             shape (B, N)    [복소수]
//   C          : IQ 포인트 집합,           shape (B, N, 2)
//   F_spec     : FFT 전력 스펙트럼,        shape (B, N/2)
//   h_C, h_F   : 인코더 특징 벡터,         shape (B, embed_dim)
//   z_C, z_F   : ℓ₂ 정규화 임베딩,        shape (B, d)
//   z_fused    : 평균 융합 임베딩,         shape (B, d)
//   S          : 코사인 유사도 행렬,      shape (B, B)
//   τ          : 학습 가능한 온도 스칼라
//   λ          : 대조 손실 가중치 스칼라
// ─────────────────────────────────────────────────────────────

PROCEDURE DDC_AMC_TrainingStep(batch_x, batch_y, model, optimizer, λ):

    // ── 1. 신호 전처리 ───────────────────────────────────────
    FOR EACH signal x IN batch_x:
        C      ← stack(Re(x), Im(x), axis=-1)       // IQ 좌표 → (N, 2)
        F_spec ← 10 · log10(|FFT(x · Hanning)|²)    // 로그 스케일 단측 PSD → (N/2,)

    // ── 2. 인코더 순전파 ─────────────────────────────────────
    h_C ← Encoder_C(C)            // Set Transformer (순열 불변)
                                   //   입력: (B, N, 2) → 출력: (B, embed_dim)
    h_F ← Encoder_F(F_spec)       // 1D CNN + GlobalAvgPool
                                   //   입력: (B, N/2)  → 출력: (B, embed_dim)

    // ── 3. Projection 및 L2 정규화 ──────────────────────────
    z_C ← L2_normalize( MLP_C(h_C) )    // Projection head g_C → (B, d)
    z_F ← L2_normalize( MLP_F(h_F) )    // Projection head g_F → (B, d)

    // ── 4. 융합 임베딩 및 분류 ──────────────────────────────
    z_fused ← (z_C + z_F) / 2           // 원소별 평균 → (B, d)
    logits  ← Classifier(z_fused)        // 2층 MLP → (B, M)

    L_cls ← CrossEntropy(logits, batch_y)

    // ── 5. CLIP 스타일 대조 손실 ───────────────────────────
    // 쌍별 코사인 유사도 행렬 구성
    FOR i = 1 TO B:
        FOR j = 1 TO B:
            S[i, j] ← dot(z_C[i], z_F[j]) / τ    // (i == j) → 양성 쌍

    // 행과 열에 대한 대칭적 교차 엔트로피
    labels ← [0, 1, 2, ..., B−1]                  // 대각선 = 양성 쌍 인덱스

    L_C→F ← CrossEntropy(S,   labels)             // C로부터 F 검색
    L_F→C ← CrossEntropy(S.T, labels)             // F로부터 C 검색

    L_contra ← (L_C→F + L_F→C) / 2

    // ── 6. 전체 손실 및 파라미터 갱신 ──────────────────────
    L_total ← L_cls + λ · L_contra

    optimizer.zero_grad()
    L_total.backward()          // 그래디언트가 두 인코더를 통해 전파됨
    optimizer.step()            // Encoder_C, Encoder_F, MLP_C, MLP_F,
                                // Classifier, τ 를 공동으로 갱신

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
    ŷ        ← argmax( Classifier(z_fused) )         // 예측 변조 클래스

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
