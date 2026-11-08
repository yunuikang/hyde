# HyDE 코드 설명서

이 문서는 `hyde/` 코드가 **무엇을, 어떤 순서로, 왜** 하는지 설명한다.
처음 읽는 사람은 §1(전체 그림) → §2(한 번 실행할 때 일어나는 일)만 읽어도 흐름을 잡을 수 있다.
나머지 절은 파일별 상세다.

---

## 1. 전체 그림

### 1-1. 모델 구조

```
            ┌────────────────────────────── 학습 (fit 80%) ───────────────────────────────┐
 flow CSV ─▶│ data.py ─▶ Encoder(PairBind) ─▶ H ∈ {±1}^D                                  │
            │                                  ├─▶ Step2: 전 클래스 분류기 M (C×D)          │
            │                                  │     MASS | AdamW (+class weight, +hard neg) │
            │                                  │     또는 SAMME boosting                     │
            │                                  └─▶ Step1: 정상 프로토타입 N (K×D)            │
            │                                        K-means → 합산 → pull/push retrain      │
            └─────────────────────────────────────────────────────────────────────────────┘
 예측:  s1 = max_k cos(H, N_k) ≥ θ1                    → benign        (Step1이 켜져 있을 때)
        아니면 c* = argmax Step2 점수
        unknown이 켜져 있고 max_c cos(H, M_c) < θ2   → unknown (−1)
        아니면 c*
```

| 기호 | 뜻 |
|---|---|
| D | 하이퍼벡터 차원(기본 10,000) |
| C | 클래스 수(benign 포함) |
| K | 정상 프로토타입 수(Step1, elbow 값 기본) |
| θ1 | Step1 정상 판정 임계(튜닝) |
| θ2 | unknown 판정 임계(val로 자동 보정) |

### 1-2. 파일 지도

| 파일 | 역할 | 주요 내용 |
|---|---|---|
| `hyde/config.py` | 설정 읽기 | `default.yaml` ← `tuned/<ds>.yaml` ← `--set` 순서로 덮어쓰기 |
| `hyde/data.py` | 데이터 | CSV → 숫자/범주 배열 + 라벨, 캐시, 층화 분할 |
| `hyde/encoders.py` | 인코더 | `PairBindEncoder`(주), `NonlinearEncoder`(비교용) |
| `hyde/model.py` | HyDE 모델 | Step2(MASS/AdamW/SAMME, CW, HN), Step1(K-means+retrain), 예측·unknown·저장 |
| `hyde/metrics.py` | 지표 | macro/weighted F1, per-class, 이진, unknown 지표 |
| `hyde/baselines.py` | ML 대조군 | xgboost, lightgbm, RF, DT, NB, MLP |
| `run.py` | 1회 실행 | 데이터 준비 → 학습 → 평가 → JSON / 모델 저장 |
| `tune.py` | 튜닝 | val 기준으로 데이터셋별 파라미터를 순차 선택 |
| `jobs.py` | 실험 단계 | E0~E5 작업 목록 생성·순차 실행·세션 분할 |
| `report.py` | 결과 표 | JSON 여러 개 → 평균 ± 표준편차 표 |
| `infer.py` | 배포용 추론 | numpy만으로 저장 모델 실행(라즈베리파이), 처리량 측정 |

**의존 방향**: `run.py`가 `hyde/*`를 쓴다. `tune.py`와 `jobs.py`는 `run.py`를 쓴다.
`infer.py`는 **아무것도 import하지 않는다**(라즈베리파이에 단독으로 복사하기 위해).

---

## 2. `python run.py --data mirai` 한 번에 일어나는 일

1. **설정 합치기** (`config.build_config`):
   `configs/default.yaml`을 읽고, `configs/tuned/mirai.yaml`이 있으면 덮어쓴다. 그 위에 `--set key=value`를 적용한다.
2. **데이터 로드** (`data.load_dataset`): `cache/mirai.npz`가 있으면 읽고, 없으면 CSV를 파싱해서 만든다(§3).
3. **분할** (`run.prepare_data`)
   - train 파일을 층화로 **fit 80% / val 20%**로 나눈다(seed 고정).
   - `--holdout`이 있으면 그 클래스를 fit/val에서 지우고, test에서는 정답을 −1(unknown)로 바꾼다.
   - fit에 남은 클래스만으로 **로컬 인덱스**(0..C−1, benign=0)를 다시 매긴다.
4. **인코더 학습** (`HydeSession`)
   - PairBind의 구간 경계와 원자를 **fit 행으로만** 정한다.
   - fit/val/test 각각을 `prepare`한다(행마다 구간 번호를 계산해 두는 단계, CPU).
5. **모델 학습** (`HyDE.fit`): Step2를 학습한 다음 Step1을 학습한다(§5).
   매 epoch val로 체크포인트를 고른다.
6. **평가** (`evaluate_hyde`)
   - val/test의 점수(s1, s2, s2cos)를 한 번 계산한다.
   - θ1(설정)과 θ2(unknown이 켜져 있으면 val로 보정)로 예측한다.
   - 지표를 계산한다. 같은 모델을 sign(프로토타입)으로도 채점해 `test_int8_proto`를 남긴다.
7. **저장**
   - `--save-model`이면 `models/<tag>/mirai__s0.npz`에 저장한다.
   - **infer.py(numpy)로 test 앞부분을 다시 예측**해 일치율(`infer_agreement`)을 기록한다.
8. **결과 JSON**: `results/<tag>/mirai__hyde__s0.json`. 이미 있으면 처음부터 건너뛴다(`--force`로 강제 실행).

ML 대조군(`--model xgboost` 등)은 4~5 대신 `baselines.py`로 학습한다. 3·6·8은 같다.

---

## 3. 데이터 — `hyde/data.py`

### 3-1. 컬럼 규칙 (11개 데이터셋 공통, `configs/datasets.yaml`)

컬럼 이름은 정규화해서 비교한다: 앞뒤 공백 제거, 소문자, 공백과 `-`는 `_`.

| 순서 | 규칙 | 예 |
|---|---|---|
| 1 | 라벨 = `attack_name`. `attack_flag`, `attack_step` 등 다른 라벨 컬럼은 드롭 | |
| 2 | **메타데이터 드롭**: 시각, IP, flow id | `window_start`, `timestamp`, `src_ip_addr`, `*_ip` |
| 3 | **누수 드롭**: 포트(15가지 표기), 날짜, nsl의 `unknown`(난이도, 라벨 파생) | `l4_src_port`, `destination_port`, `tcp.srcport` |
| 4 | 강제 범주형: `protocol`, `service`, `flag` + 데이터셋별 목록(edge-iiot, lspr23) | CSV 원문 문자열 그대로 읽음 |
| 5 | 나머지는 **train 전체를 훑어서**, 비어 있지 않은 값이 전부 float로 읽히면 숫자형. `inf`/`Infinity`/`nan`은 숫자로 인정하고 NaN 처리 | |
| 6 | 숫자형인데 train 고유값이 1개 → 드롭(상수). 범주형인데 고유값이 4096 초과 → 드롭(고카디널리티) | mirai의 backward_* 15개 |

드롭된 컬럼과 사유는 캐시의 `meta.dropped`와 모든 결과 JSON의 `data.dropped_columns`에 남는다.

### 3-2. 라벨 정리
- benign 별칭(`benign`, `benigntraffic`, `normal`, `-`, 빈칸)은 모두 `benign`으로 바꾼다.
- cicids2017: train의 `brute force`와 test의 `web attack � brute force` 같은 철자를 하나로 통일한다(`canon_cicids2017`).
  **기존 multi-tester ML 실행에서 빠져 있던 처리다.**
- test에만 있는 라벨은 −1(unknown)이 된다. 11개 중 nsl-kdd의 17종(3,750행)뿐이다.
- 클래스 번호: benign=0, 나머지는 train 빈도 내림차순.

### 3-3. 캐시 `cache/<ds>.npz`

| 배열 | 모양·타입 | 내용 |
|---|---|---|
| `X_num`, `X_num_test` | (N, Fn) float32 | 숫자 피처, 결측은 NaN |
| `X_cat`, `X_cat_test` | (N, Fc) int32 | 범주 코드. 코드 = len(vocab)이면 train에 없던 값 |
| `y`, `y_test` | (N,) int64 | 클래스 번호. test의 −1은 unknown |
| `meta` | JSON 문자열 | 클래스 이름, 컬럼 이름, 범주 사전, 드롭 내역, test-only 라벨, 클래스별 개수 |

캐시는 한 번만 만든다. 컬럼 규칙을 바꿨으면 `load_dataset(name, rebuild=True)`를 쓰거나 캐시 파일을 지운다.

### 3-4. 분할 함수
- `split_fit_val(y, 0.2, seed)`: 클래스마다 20%를 val로 보낸다(최소 1, 최대 n−1행). 1행짜리 클래스는 fit에 둔다.
- `stratified_subset(idx, y, frac=…)`: 클래스 비율을 유지하며 일부를 뽑는다. 클래스당 최소 50행을 보장한다.
  Tier B 튜닝(10%)과 PLE 트리 학습 표본에 쓴다.

---

## 4. 인코더 — `hyde/encoders.py`

두 인코더의 사용법은 같다.

```python
enc.fit(X_num_fit, X_cat_fit, cat_widths, y_fit)   # fit 행만 본다
prep = enc.prepare(X_num, X_cat)                    # 행별 상태 미리 계산 (CPU, 1회)
H = enc.encode(prep, idx)                           # idx 행들 → (B, D) int8 ±1 (GPU)
```

**(N, D) 행렬을 만들지 않는다.** 학습 루프가 배치마다 `encode`를 다시 부른다(스트리밍).
그래서 2,200만 행 데이터셋도 GPU 메모리 1 GB 안에서 돈다.

### 4-1. NonlinearEncoder (기존 HDC, 비교용)
1. 숫자 피처를 MinMax로 [0, 1]에 맞춘다(fit 행 기준, NaN은 0, 범위 밖은 clip). 범주는 one-hot이다.
2. `H = sign(cos(x · W))`, `W ~ N(0,1)`, 모양 (F, D). L2 정규화는 하지 않는다(attackhd와 같다).
- 한계: 모든 피처가 x·W에서 **먼저 섞인 뒤** 비선형이 들어간다.
  Gorishniy et al.(NeurIPS 2022)이 "하지 말라"고 지적한 형태이고, PairBind가 바로 이 점을 개선한다.

### 4-2. PairBindEncoder (주 인코더, `pairbind/pairbind.py` 이식)

**아이디어**: 피처마다 따로 ±1 "원자"를 만든다. 그다음 HV 좌표마다 피처 **두 개**를 골라 원자를 곱한다.
→ 각 좌표가 "피처 a의 상태 × 피처 b의 상태"라는 2차 상호작용을 담는다.

1. **피처 상태** (`fit` / `prepare`)
   - 숫자 피처, 고유값 > 32: **target-aware PLE**. 그 피처 하나와 라벨로 결정트리(최대 리프 16 = `bins`)를
     학습하고, 분할 지점을 구간 경계로 쓴다. 각 값은 (구간 번호 t, 구간 안 위치 α ∈ [−1, 2])가 된다.
   - 숫자 피처, 고유값 ≤ 32: 값 자체를 범주처럼 다룬다(값마다 원자 + 처음 보는 값/NaN 슬롯).
   - 범주 피처: 코드마다 원자(+ unknown 슬롯).
2. **원자**: 피처 f, 좌표 d마다 랜덤 벡터 r를 두고 `sign(rᵀφ(x))`.
   - 숫자 피처의 φ는 "중심화 온도계"(지나온 구간은 +1, 남은 구간은 −1, 현재 구간은 α만큼)다.
   - 누적합으로 O(1)에 계산한다: `2·(prefix[t] + α·r_t) − Σr`.
   - 결과가 정확히 0이면 미리 정한 랜덤 ±1(tie)을 쓴다. NaN이 이 경우다.
3. **결합**: 좌표 d마다 피처 쌍 {a, b}를 균등하게 고르고(seed 고정), `H_d = atom_a · atom_b`.
   - 피처마다 자기가 들어간 좌표만 계산하므로, 행당 연산은 F·D가 아니라 약 2D다.
- 의미: 두 샘플의 HV 내적의 기댓값 = 피처별 각도 커널들의 **2차 ANOVA 커널**. D는 정밀도를 정하고,
  상호작용 차수는 2로 고정이다.
- `prepare`는 행마다 (구간 번호, α)만 저장한다. 행렬 크기는 int16 (N×F) + float32 (N×F_num)다.

---

## 5. 모델 — `hyde/model.py`

### 5-1. Step2: 전 클래스 분류기 (benign 포함 C개)

학습 데이터는 fit 전체다. 아래 셋 중 하나를 쓴다.

**① MASS** (`readout: mass`, 기본. multi-tester `retrain_mass` / attackhd와 같은 식)

```
M ← 0                                   (C×D, zero-init)
매 epoch: 순서 = 무작위 셔플 (hard negative가 켜지면 재샘플링, 아래)
  배치(8192)마다:
     S = H Mᵀ                           (내적)
     P = softmax(relu(S))
     R = onehot(y) − P                   (정답 클래스는 당기고, 오답 클래스는 민다)
     R ← R ⊙ w[y] / mean(w[y])           (class weight, 배치 안에서 평균 1로 재정규화)
     M ← M + lr · Rᵀ H                   (lr 기본 0.035)
```

**② AdamW** (`readout: adamw`, pairbind의 `ClassHVReadout`)
- `Linear(D, C)`의 가중치를 L2 정규화한 클래스 평균 HV로 초기화한다.
- `CrossEntropy(H/√D · Wᵀ + b, y, weight=w)`를 AdamW(lr 0.01, wd 1e-4)로 학습한다.
- MASS와 gradient 형태가 같고 옵티마이저만 다르다. 튜닝(E1 0단계)에서 둘 중 하나를 고른다.

**③ SAMME boosting** (`boosting.T > 0`이면 ①②를 대체)
- 라운드마다 샘플 가중치 D_t로 클래스별 가중합 프로토타입 C_t를 만든다(약한 학습기).
- 오분류율 ε, α_t = log((1−ε)/ε) + log(C−1), D_t 갱신(상한 100/N).
- ε ≥ 1 − 1/C이면 멈춘다. 점수 = Σ α_t·cos(H, C_t) / Σα.
- class weight가 켜져 있으면 초기 D_1을 클래스 가중치에 비례하게 둔다.

**정상 하위 프로토타입** (`step2.benign_split = k`, k > 1일 때)
- 정상 트래픽에는 여러 유형(웹, DNS, 스트리밍 …)이 섞여 있어 benign 벡터 하나로는 평균이 되어 버린다.
- 학습 전에 fit의 benign HV를 K-means로 k개로 나눠 **benign_1 … benign_k를 서로 다른 클래스로 학습**하고,
  예측할 때 이들 점수의 최댓값을 benign 점수로 합친다(`split_benign`, `_merge_scores`).
- Step1 게이트와 달리 Step2의 판단을 덮어쓰지 않고 **분류기의 표현력만 늘린다.**
- class weight는 원래 클래스 기준으로 계산하고 하위 클래스가 benign의 가중치를 물려받는다.
  hard negative와 SAMME에서는 benign 하위 클래스끼리 헷갈리는 것을 오답으로 치지 않는다.
- 저장 파일에 `step2_merge`(하위 클래스 → 원래 클래스)가 들어가고 infer.py도 같은 방식으로 합친다.

**class weight** (`class_weight.p`): `w_c = min((N/(C·n_c))^p, 100)`을 샘플 평균이 1이 되게 정규화한다. p=0이면 끈다.

**hard negative** (`hard_negative.beta`, Robinson 2021)
- 매 epoch 시작에 fit 전체에 대해 `h_i = max_{c≠y} cos(H_i, M_c) − cos(H_i, M_y)`를 계산한다
  (헷갈리는 샘플일수록 큼).
- `p_i ∝ exp(β·h_i)`로 N개를 복원추출해 그 epoch의 순서로 쓴다. β=0이면 균등 셔플이다.
- 항상 코사인을 쓰는 이유: 내적/로짓은 범위가 없어 exp가 폭발한다(pairbind에서 실제로 붕괴를 확인).
- 실제로 몇 %의 행이 뽑혔는지가 `history.hn_unique_fraction`에 남는다(균등이면 약 63%).

**체크포인트 선택**
- 매 epoch 끝에 val(최대 20만 행, 희귀 클래스는 전부 포함)에서 `select_metric`(기본 macro-F1)을 잰다.
- 최고 시점의 M을 쓴다. 최대 60 epoch, 8 epoch 동안 개선이 없으면 중단한다.
- epoch별 기록은 `history.step2_epochs`, 선택된 epoch은 `history.selected_epoch`.

### 5-2. Step1: 정상 프로토타입 (선택)
1. fit의 benign HV 중 최대 5,000개로 K-means(K개, seed 고정)를 학습한다.
2. benign 전체를 가장 가까운 중심에 배정하고, 클러스터별로 HV를 합해 N_k를 만든다(singlepass).
3. **retrain** (multi-tester `retrain_anomaly`, GenieHD에서 착안). 5 epoch, lr 0.05, 배치 512
   - 정상 S: 가장 가까운 N_k ← N_k + lr·(1 − cos)·S    (잘 맞는 샘플은 조금만 당긴다)
   - 공격 S: 모든 N_k ← N_k − lr·max(cos, 0)·S         (가까운 프로토타입일수록 세게 민다)
4. 점수 `s1 = max_k cos(H, N_k)`. `s1 ≥ θ1`이면 benign으로 판정한다.

### 5-3. 예측과 unknown
- `scores()`가 한 번에 세 가지를 계산한다.
  - s1(Step1)
  - s2(Step2 점수: MASS는 내적, AdamW는 로짓, SAMME는 가중 코사인)
  - s2cos(클래스 프로토타입과의 최대 코사인)
- `predict(scores, θ1, θ2)`는 **numpy만 쓰는 정적 함수**다. 임계를 바꿔도 재학습이 필요 없다.
  tune.py의 θ1 스윕이 이걸 쓴다.
- θ2 보정(`calibrate_theta2`): val에서 **정답이 공격**이고 Step1을 통과하지 못한 행들의 s2cos 분포에서
  `reject_rate`(기본 5%) 분위수를 θ2로 둔다. 즉 known 공격의 5%만 unknown으로 잘못 거르는 수준이다.
  test는 보지 않는다.

### 5-4. 저장 형식 (`save`)
`.npz` 하나에 다음을 담는다.

| 키 | 내용 |
|---|---|
| `enc_*` | 인코더 상태. PairBind: 쌍 인덱스 `enc_pairs`, 피처별 `enc_r{f}_kind/source/width/coords/weights/ties/edges/categories`. Nonlinear: `enc_lo/span/W/cat_widths` |
| `step1_N` | (K, D) |
| `step2_M`, `step2_bias` | (C, D), AdamW이면 bias. SAMME이면 `samme_T` (T, C, D)와 `samme_alpha` |
| `meta` (JSON) | 클래스 이름, θ1, θ2, readout, 인코더 종류, D, 전체 config, **전처리 정보**(숫자/범주 컬럼 이름, 범주 사전, 라벨 컬럼, benign 별칭, cicids2017 canon 여부) |

전처리 정보가 들어 있으므로 `infer.py`는 원본 CSV를 바로 읽어 같은 피처를 만든다.

---

## 6. 지표 — `hyde/metrics.py`

| 지표 | 정의 |
|---|---|
| `macro_f1`, `weighted_f1` (동등 주 지표) | sklearn, `labels = test에 실제로 있는 클래스` (benign 포함, nsl의 unknown(−1) 포함) |
| `macro/weighted_precision/recall`, `accuracy` | 같은 labels |
| `macro_f1_attack_only` | benign을 뺀 macro |
| `per_class` | 클래스별 P, R, F1, support (test에만 저장) |
| `binary` | benign 대 나머지로 접은 TP/FP/TN/FN, F1(공격), recall, precision, **FPR**, 이진 macro-F1. 예측 −1은 공격으로 센다 |
| `unknown` 블록 (unknown 실험) | unknown recall / precision / F1, **AUROC**(s2cos 기준, known 공격 대 unknown), known 클래스만의 macro/weighted-F1, benign → unknown 오탐률 |
| `fast_f1` | 학습 중 val 계산용 빠른 버전(혼동행렬 기반) |

---

## 7. ML 대조군 — `hyde/baselines.py`

- **HyDE와 같은 조건**: 같은 캐시, 같은 드롭 컬럼, 같은 fit/val/test, 같은 지표 함수.
- 입력: MinMax(숫자) + one-hot(범주) dense float32 행렬(`FeatureMatrix`).
- 하이퍼파라미터는 multi-tester 설정 그대로다.

  | 모델 | 설정 |
  |---|---|
  | xgboost | 300, lr 0.05, hist |
  | lightgbm | 100, lr 0.1, deterministic, col_wise |
  | RF | 100 |
  | DT | 기본 |
  | NB | Gaussian |
  | MLP | PyTorch 3×512, Adam 1e-3, batch 256, 50 epoch, val macro-F1 체크포인트 |

- **class weight**: xgboost, lightgbm, RF, MLP는 p ∈ {0, 0.5} 두 번 학습하고 val macro-F1이 높은 쪽을 쓴다.
  `sample_weight` 또는 CE weight로 넣는다. 선택 결과는 JSON `ml.class_weight_p`.
- unknown: 최대 예측 확률이 θ보다 낮으면 −1. θ는 HyDE와 같은 방식으로 val에서 정한다.
- `--measure-size`: pickle 크기 = 모델 바이트.

---

## 8. 튜닝 — `tune.py`

데이터셋마다 **val만 보고** 파라미터를 한 축씩 고른다. 각 축에서 최고값을 고정하고 다음으로 넘어간다.
동점이면 먼저 나온(더 단순한) 값을 유지한다.

| 단계 | 축 | 후보 | 비고 |
|---|---|---|---|
| A | readout | mass, adamw | Step1을 끈 상태 |
| A | lr | mass {0.035, 0.01, 0.1} / adamw {0.01, 0.003, 0.03} | |
| A | class weight p | 0, 0.25, 0.5, 0.75, 1.0 | |
| A | hard negative β | 0, 0.5, 1, 2, 4 | |
| A | boosting T | 0, 5, 10, 20 | |
| A | 정상 하위 프로토타입 수 | 0(안 나눔), elbow K, 2K | 2026-11-08 추가. 이전에 튜닝된 데이터셋은 이 축만 추가 튜닝(확장 모드) 후 Step1 재확인 |
| B | Step1 | 끔 / K ∈ {1, elbow, 2×elbow} × retrain {끔, lr 0.01, lr 0.05} × θ1 {0.70…0.97} | Step2는 한 번만 학습, θ1은 사후 스윕 → 조합 전체를 싸게 비교 |
| C | PairBind bins | 16, 8, 32 | 전체 파이프라인 |

- 인코더가 같은 후보끼리는 인코더와 `prepare` 결과를 재사용한다(`Tuner.session`).
- Tier B(대형 3개)는 fit을 10%로 줄여 튜닝한다.
- 결과: `configs/tuned/<ds>.yaml`(바뀐 키만), 전 과정 기록은 `results/tune/<ds>/log.json`.
- 이미 tuned 파일이 있으면 건너뛴다(다시 하려면 파일을 지운다).

---

## 9. 실험 실행기 — `jobs.py`

| 단계 | 작업 | 결과 위치 |
|---|---|---|
| E0 | ML 6종 × 11 × 3시드 | `results/ml/` |
| E1 | `tune.py` × 11 | `configs/tuned/` |
| E2 | 튜닝 구성 × 3시드 + 모델 저장 | `results/final/`, `models/final/` |
| E3 | ablation: 튜닝 구성에서 **구성요소 하나만 뒤집는다**(`ablation_arms`) | `results/ablation/<arm>/` |
| E4 | unknown: 데이터셋마다 큰/중간/작은 공격 클래스(test ≥ 30행)를 각각 holdout. 2-step / 1-step / RF / xgboost 비교 | `results/unknown/<variant>/<scenario>/` |
| E5 | D ∈ {1000, 2000, 5000}, seed 0, Tier A, 모델 저장(크기 측정) | `results/e5/D<d>/` |

E3 ablation 팔의 규칙
- `encoder_nonlinear`: 인코더만 nonlinear로 바꾼다.
- class weight / hard negative / boosting / 정상 하위 프로토타입: 튜닝에서 켜졌으면 끄고, 꺼졌으면 켠다(p=0.5, β=1, T=10, k=elbow K).
- Step1이 튜닝에서 켜졌으면 `step1_off`, `step1_K1`, `step1_retrain_*`.
  꺼졌으면 `step1_on`(θ1=0.95)으로 "켰을 때"를 보인다.

E4 세부
- 시나리오는 처음 실행할 때 `results/unknown/scenarios.csv`에 고정된다. 다시 뽑지 않는다.
- nsl-kdd는 test-only 17종을 쓰는 `natural` 시나리오가 추가된다.

공통 동작
- 결과 JSON이 이미 있는 작업은 `run.py`가 건너뛴다. 중단한 뒤 같은 명령을 다시 돌리면 이어서 진행된다.
- `--shard i/n`: 작업 목록의 i, i+n, i+2n, … 번째만 실행한다. 세션 3개라면 각 세션에서 `0/3`, `1/3`, `2/3`을 돌린다.
  `results/`는 공유 폴더라 그대로 합쳐진다.

---

## 10. 결과 JSON (`results/<tag>/<ds>__<model>__s<seed>.json`)

| 키 | 내용 |
|---|---|
| `test`, `val` | §6 지표 (`test`에만 per_class) |
| `unknown`, `theta1`, `theta2` | unknown 실험 지표와 사용한 임계 |
| `test_int8_proto` | 같은 모델을 sign(프로토타입)으로 채점한 macro/weighted/acc (HyDE) |
| `history` | step2 epoch 기록, 선택 epoch, hard negative 샘플링 비율, SAMME 라운드, step1 K, 단계별 초 |
| `ml` | ML: class weight 후보별 val 점수와 선택된 p |
| `train_sec`, `encoder_sec`, `eval_sec`, `wall_sec`, `infer_ms_per_sample` | 시간 |
| `peak_rss_mb`, `peak_gpu_mb`, `model_bytes` | 메모리와 모델 크기 |
| `infer_agreement` | infer.py(numpy)와 학습 코드 예측의 일치율 (`--save-model`일 때) |
| `config`, `git`, `data` | 실행에 쓴 전체 설정, 코드 버전, 데이터 요약(행 수, 피처 수, 드롭 컬럼, 클래스) |

`report.py`가 이 파일들을 (tag, model) 단위로 묶어 데이터셋별 평균 ± 표준편차 표를 만든다.

---

## 11. 배포용 추론 — `infer.py` (라즈베리파이)

```bash
# 라즈베리파이에는 infer.py와 모델 .npz만 복사한다 (numpy 필요)
python3 infer.py --model mirai__s0.npz --csv test-flow.csv --bench
python3 infer.py --model mirai__s0.npz --csv test-flow.csv --int8-proto --bench    # 경량 모드
python3 infer.py --model mirai__s0.npz --csv test-flow.csv --repeat-seconds 120   # 전력 측정용 반복
```

- CSV 파싱 → 전처리(저장된 컬럼·사전 사용) → PairBind/Nonlinear 인코딩 → 점수 → θ1/θ2 예측.
  모두 numpy로만 한다. 학습 코드(torch)와 같은 수식이고, run.py가 저장할 때마다 일치율을 확인한다.
- 출력
  - 라벨이 있으면 macro/weighted-F1, accuracy
  - `--bench`: 처리량(flows/s), 배치 지연 p50/p95, 모델 크기, peak RSS. CSV 파싱 시간은 따로 표시한다.
- **전력 측정 절차**
  1. 아무것도 돌리지 않은 상태의 전력(W_idle)을 1분간 잰다.
  2. `--repeat-seconds 120`으로 추론을 반복하는 동안의 평균 전력(W_run)을 잰다. USB 전력계나 INA219를 쓴다.
  3. 출력된 처리량으로 계산한다: flow당 에너지 = (W_run − W_idle) ÷ 처리량 [J/flow].

---

## 12. 설정 키 (`configs/default.yaml`)

| 키 | 기본 | 설명 |
|---|---|---|
| `seed` | 0 | 분할·인코더·쌍 추출·K-means·학습 순서·ML 전부 |
| `threads` / `device` | 6 / cuda | CPU 스레드, 연산 장치 |
| `data.val_frac` | 0.2 | fit/val 비율 |
| `data.tune_subsample` | null | 튜닝용 fit 축소 비율 |
| `encoder.type` | pairbind | pairbind / nonlinear |
| `encoder.D`, `encoder.bins` | 10000, 16 | 차원, PLE 최대 구간 수 |
| `step1.enabled`, `K`, `theta1` | true, null(=elbow), 0.90 | Step1 |
| `step1.retrain.*` | true, 5 epoch, lr 0.05, batch 512 | pull/push |
| `step2.readout` | mass | mass / adamw |
| `step2.mass.lr`, `adamw.lr` | 0.035 / 0.01 | |
| `step2.epochs`, `patience`, `select_metric`, `val_select_max` | 60, 8, macro_f1, 200000 | 체크포인트 선택 |
| `step2.benign_split` | 0 | > 1이면 benign을 그 수만큼 하위 클래스로 나눠 학습 |
| `class_weight.p`, `cap` | 0 (끔), 100 | |
| `hard_negative.beta` | 0 (끔) | |
| `boosting.T`, `d_cap_mult` | 0 (끔), 100 | |
| `unknown.enabled`, `reject_rate` | false, 0.05 | |
| `eval.int8_proto`, `infer_check_rows` | false, 20000 | |
| `ml.class_weight_p`, `ml.mlp.*` | [0, 0.5], 3×512 등 | ML 대조군 |

명령줄에서 덮어쓰기: `--set step2.readout=adamw --set class_weight.p=0.5`. 없는 키를 쓰면 에러가 난다.

---

## 13. 어디서 가져왔나 (출처 대응)

| hyde 코드 | 원본 |
|---|---|
| `data.py` 누수 컬럼 목록, cicids2017 라벨 통일, 전체 스캔 타입 판정 | `pairbind/data.py` (`LEAKY_COLUMNS`, `canonical_label`, `scan_columns`) |
| `data.py` benign 별칭 | `multi-tester/modules/dataset_handler.py` |
| 데이터셋별 K, 범주 컬럼 목록 | `multi-tester/scripts/_common.sh`, `analysis/elbow/summary.csv` |
| `PairBindEncoder` | `pairbind/pairbind.py` (`PairBindEncoder`: 같은 난수 스트림·같은 수식) |
| `NonlinearEncoder` | `attackhd/hdc.py` `NonlinearEncoder`, `multi-tester/algorithms/hdc.py` `encode_nonlinear` |
| MASS, class weight, hard negative, SAMME, Step1 retrain | `multi-tester/algorithms/hdc.py` (`retrain_mass`, `_compute_class_weights`, `_compute_hardness_weights`, `retrain_boost_samme`, `retrain_anomaly`) |
| AdamW readout | `pairbind/pairbind.py` `ClassHVReadout` (refit 대신 val 체크포인트) |
| ML 하이퍼파라미터 | `multi-tester/algorithms/*.py`, `SWEEP_RESULTS.md` §6 |

---

## 14. 확인된 것과 주의할 점

### 개발 중 확인 (2026-11-07)

| 항목 | 결과 |
|---|---|
| 데이터 로더 | mirai, nsl-kdd, cicids2017, edge-iiot 캐시 생성. nsl의 test-only 17종(3,750행) 검출. cicids2017 웹 공격 통일 후 test-only 0 |
| HyDE 경로 | MASS, AdamW, class weight, hard negative, SAMME, nonlinear, Step1 K=1, unknown holdout 모두 mirai에서 관통 |
| infer.py 일치율 | 저장 모델의 numpy 추론이 학습 코드 예측과 **100% 일치** (cicids2017 포함). SAMME만 99.989% — 라운드 점수 합산의 부동소수점 반올림으로 동점 근처 1행이 갈림 |
| ML 6종 | mirai 관통, class weight 선택 동작 |
| tune.py | mirai 전체 튜닝 2분 |
| 속도 | cicids2017(fit 158만 행): epoch당 약 5초, 1회 약 100초, GPU 0.7 GB |

### 주의
- **디스크**: `/home/work`가 거의 가득 차 있다(남은 공간 약 95 GB). 대형 3개 캐시가 합쳐서 약 15 GB 들어간다.
- `nproc`은 1로 나오지만 실제 사용 가능한 코어는 6개다(`os.sched_getaffinity`). 기본 `threads: 6`.
- E4 2-step은 튜닝에서 Step1이 꺼진 데이터셋이면 `step1.enabled=true` + tuned θ1(기본 0.90)로 켠다.
- 대형 데이터셋 ML(특히 RF)은 6코어에서 데이터셋당 1시간 이상 걸릴 수 있다. GPU 작업과 다른 세션에서 돌리기를 권한다.
- `infer.py`의 CSV 파서는 순수 파이썬이라 수백만 행이면 파싱이 느리다. 처리량 측정은 인코딩+점수 구간만 잰다.

## 15. 자주 하는 작업

| 하고 싶은 것 | 방법 |
|---|---|
| 데이터셋 추가 | `configs/datasets.yaml`에 한 줄(dir, K, tier, 필요하면 categorical). CSV는 `training-flow.csv`/`test-flow.csv`, 라벨 컬럼 `attack_name` |
| 튜닝 후보 바꾸기 | `tune.py` 상단 `GRID` |
| 다시 튜닝 | `configs/tuned/<ds>.yaml` 삭제 후 `python tune.py --data <ds>` |
| 결과 다시 만들기 | 해당 JSON 삭제 또는 `run.py --force` |
| 컬럼 규칙 변경 후 | `cache/<ds>.npz` 삭제(다음 실행 때 재생성) |
| 한 실행의 설정 확인 | 결과 JSON의 `config` |
