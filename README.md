# HyDE — HDC 기반 경량 네트워크 침입 탐지 (최종 테스터)

코드 구조와 동작 원리는 **[ARCHITECTURE.md](ARCHITECTURE.md)**, 실험 계획은 `../plans/EXPERIMENT_PLAN.md`.

## 설치
서버에는 이미 있음: python 3.10, torch 1.12, numpy, polars, scikit-learn, xgboost, lightgbm, pyyaml.
라즈베리파이: `numpy`만 있으면 `infer.py`가 돈다.

## 가장 빠른 사용법
```bash
cd /home/work/yunuikang/hyde
python run.py --data mirai                         # HyDE 1회 (기본 설정, 결과 JSON 생성)
python run.py --data mirai --model xgboost         # ML 대조군 1회
python tune.py --data mirai                        # 데이터셋별 튜닝 → configs/tuned/mirai.yaml
python run.py --data mirai --tag final --save-model --seed 0   # 튜닝값으로 학습 + 모델 저장
python infer.py --model models/final/mirai__s0.npz \
    --csv ../multi-tester/data/mirai/test-flow.csv --bench     # numpy 전용 추론 + 처리량
```
처음 쓰는 데이터셋은 CSV를 한 번 파싱해 `cache/<ds>.npz`를 만든다(대형은 수 분~수십 분).

## `run.py` 실행 옵션

한 번 실행 = 학습 + 평가 1회. 설정은 `configs/default.yaml` → `configs/tuned/<데이터셋>.yaml`(튜닝으로 확정된 값, 자동 적용) → `--set` 순서로 덮어쓴다.
실제로 쓰인 설정 전체는 결과 JSON의 `config`에 기록된다.

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--data` | 데이터셋 이름 (`configs/datasets.yaml`의 11개 중 하나) | 필수 |
| `--model` | `hyde` 또는 ML 대조군 `xgboost` · `lightgbm` · `random_forest` · `decision_tree` · `naive_bayes` · `mlp` | `hyde` |
| `--seed` | 시드 (fit/val 분할, 인코더 난수, 학습 순서, K-means, ML 모두) | 0 |
| `--set 키=값` | 설정 항목 하나를 덮어씀 (여러 번 사용 가능, 없는 키는 오류) | — |
| `--config 파일.yaml` | 설정 파일을 통째로 추가로 덮어씀 | — |
| `--no-tuned` | 튜닝 결과를 쓰지 않고 기본값으로 실행 (`--model hyde`는 튜닝 파일이 없으면 이 옵션 없이는 실행 거부) | 튜닝값 사용 |
| `--holdout "클래스명"` | 해당 공격 클래스를 학습에서 빼고, test에서는 unknown(−1) 정답으로 평가 (쉼표로 여러 개) | — |
| `--save-model` | 모델을 `models/<tag>/<데이터셋>__s<seed>.npz`로 저장하고 numpy 추론기(`infer.py`)와 예측 일치율 확인 | 저장 안 함 |
| `--bench-cpu N` | 저장 모델로 test N행을 numpy(CPU)로 추론해 속도 측정 (float / int8) | — |
| `--measure-size` | ML 모델의 직렬화 크기 측정 | — |
| `--tag` | 결과 폴더 이름 → `results/<tag>/<데이터셋>__<모델>__s<seed>.json` | `dev` |
| `--threads` | CPU 스레드 수 | 6 |
| `--force` | 같은 결과 파일이 있어도 다시 실행 (없으면 건너뜀) | — |

### `--set`으로 바꿀 수 있는 설정 (`configs/default.yaml`)

| 구분 | 키 (기본값) | 설명 |
|---|---|---|
| 공통 | `seed` (0), `threads` (6), `device` (cuda) | `device=cpu`로 GPU 없이 실행 |
| 데이터 | `data.val_frac` (0.2) | train 중 val 비율 (층화) |
| | `data.tune_subsample` (null) | 튜닝 때 fit 일부만 사용 (대형 데이터셋 0.1) |
| 인코더 | `encoder.type` (pairbind) | `pairbind` 또는 기존 HDC `nonlinear` = sign(cos(xW)) |
| | `encoder.D` (10000) | 하이퍼벡터 차원 |
| | `encoder.bins` (16) | PLE 최대 구간 수 |
| | `encoder.order` (2) | 2 = 피처쌍 결합(PairBind), 1 = 결합 없이 PLE 원자만 |
| | `encoder.categorical_threshold` (32) | 고유값이 이 이하인 숫자 피처는 범주처럼 처리 |
| Step1 정상 게이트 | `step1.enabled` (true) | 정상 프로토타입 게이트 사용 여부 |
| | `step1.K` (null = elbow 값) | 정상 K-means 클러스터 수 |
| | `step1.theta1` (0.90) | 이 유사도 이상이면 benign으로 판정 |
| | `step1.retrain.enabled` (true), `.lr` (0.05), `.epochs` (5), `.batch` (512) | pull/push 재학습 |
| Step2 분류기 | `step2.readout` (mass) | `mass` 또는 `adamw` |
| | `step2.mass.lr` (0.035), `step2.mass.batch` (8192) | MASS 학습률·배치 |
| | `step2.adamw.lr` (0.01), `.weight_decay` (1e-4), `.batch` (4096) | AdamW 학습률·배치 |
| | `step2.epochs` (60), `step2.patience` (8) | 최대 epoch, 조기 종료 |
| | `step2.select_metric` (macro_f1) | 체크포인트 선택·튜닝 기준 (`macro_f1` / `weighted_f1`) |
| | `step2.val_select_max` (200000) | epoch 선택용 val 최대 행 수 (희귀 클래스는 전부 포함) |
| | `step2.benign_split` (0) | > 1이면 benign을 그 수만큼 하위 클래스로 나눠 학습 후 합침 |
| 학습 기법 | `class_weight.p` (0 = 끔), `class_weight.cap` (100) | 클래스 가중 `(N/(K·n_c))^p`, 상한 |
| | `hard_negative.beta` (0 = 끔) | 헷갈리는 샘플을 `exp(β·hardness)`로 더 자주 뽑음 |
| | `boosting.T` (0 = 끔), `boosting.d_cap_mult` (100) | SAMME 부스팅 라운드 수 (켜면 MASS/AdamW 대신 사용) |
| unknown 탐지 | `unknown.enabled` (false) | 어떤 클래스와도 충분히 닮지 않으면 unknown(−1) |
| | `unknown.reject_rate` (0.05) | θ2 = val의 known 공격 중 이 비율만 unknown으로 거르는 값 (결과에는 1/5/10%가 모두 기록됨) |
| 평가 | `eval.int8_proto` (false) | 프로토타입을 부호로 줄여 평가 (경량 배포 실험) |
| | `eval.infer_check_rows` (20000) | `--save-model` 때 infer.py 일치율을 확인할 test 행 수 |
| ML 대조군 | `ml.class_weight_p` ([0.0, 0.5]) | xgboost·lightgbm·RF·MLP는 이 중 val이 좋은 값 선택 |
| | `ml.mlp.hidden` (512), `.layers` (3), `.lr` (0.001), `.batch` (256), `.epochs` (50), `.patience` (8) | MLP 구조·학습 |

### 예시
```bash
python run.py --data nsl-kdd                                        # 튜닝값 그대로
python run.py --data nsl-kdd --seed 1 --save-model                 # 시드 변경 + 모델 저장
python run.py --data nsl-kdd --set encoder.type=nonlinear          # 기존 HDC 인코더와 비교
python run.py --data nsl-kdd --set encoder.D=1000 --set step1.enabled=true --set step1.theta1=0.95
python run.py --data nsl-kdd --holdout neptune --set unknown.enabled=true   # unknown 실험
python run.py --data nsl-kdd --model xgboost                       # ML 대조군
python run.py --data nsl-kdd --no-tuned --set device=cpu           # 튜닝 전 기본값, CPU 실행
```

## 세션 3개로 나눠 실행 (2026-11-08~)

| 세션 | 명령 | 맡는 일 |
|---|---|---|
| 1 | `nohup bash run_after_A.sh >> logs/after_A.log 2>&1 &` | Tier A 8개 E1 → E5, 끝나면 Lane B(역순)·ML(역순) 보조 |
| 2 | `nohup bash run_lane.sh B 6 >> logs/lane_B.log 2>&1 &` | Tier B 3개(cicids2018, lspr23, nf-ton): E1 → E5 |
| 3 | `nohup bash run_lane.sh ML <코어수> >> logs/lane_ML.log 2>&1 &` | E0 ML 대조군 전부 |
| 마지막 | `nohup bash run_lane.sh COST 6 >> logs/lane_COST.log 2>&1 &` | E6 비용 측정 — **다른 작업이 모두 끝난 뒤, 한 세션에서 단독으로** |

각 세션에서 먼저 `cd /home/work/yunuikang/hyde`. 같은 lane을 두 세션에서 동시에 돌리지 않는다.
논문의 시간·처리량·모델 크기 수치는 E6에서만 가져온다(다른 단계는 작업이 겹쳐 측정이 오염됨).

## 세션이 끊겼을 때 / 이어서 하기

전체 실험은 `run_all.sh` 하나로 돈다. **언제 다시 실행해도 안전하다** — 끝난 결과는 건너뛰고 남은 것만 돈다.
```bash
cd /home/work/yunuikang/hyde
python status.py                                             # 단계별 완료/계획 수, 실행 중 프로세스, 각 로그 마지막 줄
nohup bash run_lane.sh <A|B|ML> <threads> >> logs/lane_<X>.log 2>&1 &   # 해당 세션의 lane (재)시작
```
- 이미 돌고 있는지는 `python status.py`의 `running:`으로 확인한다(두 개를 동시에 띄우면 같은 작업을 중복 실행).
- 결과·캐시·모델·튜닝 기록은 임시 파일에 쓴 뒤 이름을 바꿔 저장한다 → 쓰는 도중 끊겨도 깨진 파일이 남지 않는다.
- 튜닝(`tune.py`)은 끝난 후보를 `results/tune/<ds>/log.json`에 남기므로 중간에 끊겨도 그 지점부터 이어서 한다.
- 진행 순서: CPU 줄 = E0(ML) / GPU 줄 = E1 튜닝 → E2 최종 → E3 ablation → E4 unknown → E5 차원.
- 로그: `logs/run_all.log`(단계 시작·종료), `logs/E*.log`(각 실행 출력).

## 실험 단계 (EXPERIMENT_PLAN.md)
```bash
python jobs.py E0            # ML 대조군 6종 × 11 데이터셋 × 3시드     → results/ml/
python jobs.py E1            # 튜닝                                   → configs/tuned/
python jobs.py E2            # 최종 HyDE × 3시드 (+모델 저장)          → results/final/, models/final/
python jobs.py E3            # ablation                               → results/ablation/<arm>/
python jobs.py E4            # unknown (큰/중간/작은 클래스 holdout)   → results/unknown/...
python jobs.py E5            # 차원 D 1000/2000/5000                  → results/e5/
```
- `--dry`: 명령만 출력 · `--datasets a b`: 일부만 · `--shard i/n`: 세션 n개가 나눠 실행
- 결과 JSON이 이미 있으면 건너뛴다 → 중단 후 같은 명령으로 재개
- 오래 걸리므로 `nohup python jobs.py E0 > logs_E0.txt 2>&1 &` 형태로 돌린다
- E2~E5는 E1(튜닝)이 끝난 데이터셋에만 돌릴 수 있다

## 결과 표
```bash
python report.py final ml                                        # 표 A: HyDE vs ML (macro / weighted F1)
python report.py "ablation/*" final                              # 표 B: ablation
python report.py "unknown/*/*" --block unknown --metrics unknown_recall unknown_precision auroc known_macro_f1
python report.py "e5/*" final --extra model_bytes infer_ms_per_sample
```
`results/reports/`에 md·csv로도 저장된다.

## 폴더
```
hyde/            라이브러리 (data, encoders, model, baselines, metrics, config)
run.py           1회 실행        tune.py   튜닝        jobs.py   실험 단계 실행기
report.py        결과 표         infer.py  라즈베리파이용 독립 추론기
configs/         default.yaml · datasets.yaml · tuned/<ds>.yaml
cache/ results/ models/          (생성물, git 제외)
```
