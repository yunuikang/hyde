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
