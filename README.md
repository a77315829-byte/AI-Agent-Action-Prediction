# DACON AI Agent Action Prediction

대화 문맥, 최근 도구 사용 이력, 열린 파일 정보와 구조화 특징을 기반으로
AI 에이전트가 다음에 수행할 행동을 14개 클래스 중 하나로 예측하는 프로젝트입니다.

## 최종 성과

- 평가 지표: Macro-F1
- 학습 데이터: 70,000개
- 검증: 세션 단위 StratifiedGroupKFold
- 최종 Public Macro-F1: **0.7893958473**
- 최종 모델: Qwen2.5-Coder-0.5B 기반 구간별 분류기 + OOF teacher distillation + LightGBM log-probability blend

## 최종 모델 구성

1. `current`, `action`, `history`, 전체 문맥을 분리해 pooling
2. handcrafted structured feature MLP 결합
3. 14개 action head와 상위 family head 공동 사용
4. OOF teacher soft target 기반 distillation
5. LightGBM 확률을 Tree weight `0.15`로 log-probability blend
6. 최종 후처리:
   - action temperature: `1.4`
   - family weight: `1.25`
   - prior beta: `-0.85`

## 저장소 구조

```text
src/
  core/              핵심 Qwen 학습·추론 코드
  inference/         제출 및 패키징 코드
  tree/              LightGBM 및 Tree blend 코드
experiments/
  model_evolution/   V13, V14, long-context, 1.5B 등
  ensembles/         teacher, soup, gated, three-way blend
  specialists/       혼동 클래스 specialist
  sequence/          transition 및 Viterbi
  ablations/         후처리, token budget, classwise weight
  retrieval/         TF-IDF 및 retrieval 계열
analysis/             집계 분석 코드와 공개 가능한 결과
configs/              최종 설정과 대표 실험 설정
results/              주요 실험 점수 표
docs/                 결과 보고서와 실패 분석
```

## 주요 결과

| 실험 | Public Macro-F1 | 결론 |
|---|---:|---|
| V23 Qwen + Tree | 0.7870583827 | 이전 기준선 |
| Rebuilt cur160 | 0.7888347480 | 원본 cur160 미달 |
| Final cur160 | **0.7893958473** | 최종 선택 |
| Teacher 50:50 blend | 0.7893696524 | 거의 동점 |
| Tree weight 0.10 | 0.7890965708 | 하락 |
| Tree weight 0.20 | 0.7886778450 | 하락 |
| V27 gated Tree | 0.7881191369 | 하락 |

세부 검증 결과와 폐기 사유는 다음 문서를 참고합니다.

- [`docs/RESULT_REPORT.md`](docs/RESULT_REPORT.md)
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)
- [`docs/INVALID_EXPERIMENTS.md`](docs/INVALID_EXPERIMENTS.md)
- [`results/experiment_summary.csv`](results/experiment_summary.csv)

## 검증 원칙

샘플 ID의 `-step_` 앞부분을 세션 ID로 사용하고 동일 세션이 학습과 검증에
동시에 포함되지 않도록 분리했습니다.

```python
groups = np.asarray([
    str(sample["id"]).rsplit("-step_", 1)[0]
    for sample in samples
])

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

## 데이터와 모델 파일

대회 데이터, 모델 체크포인트, LoRA adapter, logits, 제출 ZIP은 저장소에 포함하지 않습니다.

```text
data/
model/
*.pt
*.safetensors
*.npy
*.npz
*.zip
```

대회 데이터의 사용 및 재배포는 원 대회 규정을 따라야 합니다.

## 실행 환경

기존 프로젝트의 `requirements.txt`를 복사해 사용합니다. CUDA, PyTorch,
Transformers, PEFT, LightGBM 버전은 최종 실행 환경과 맞춰야 합니다.

Windows PowerShell 예시:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 한계

- full student에 대한 독립적인 5-fold OOF를 모두 구축하지 못했습니다.
- Teacher OOF와 full student 사이에 분포 차이가 있었습니다.
- 일부 후처리는 OOF에서 개선됐지만 실제 제출 환경에서는 재현되지 않았습니다.
- 1.5B 모델은 검증 성능과 제출 용량 측면에서 채택하지 못했습니다.
- 최종 점수는 목표 0.794에 도달하지 못했으며, 추가 미세 후처리의 기대효과는 제한적이었습니다.
