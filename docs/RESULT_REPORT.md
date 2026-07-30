# 결과 보고서

## 1. 프로젝트 개요

본 프로젝트의 목적은 대화 기록과 도구 사용 상태를 기반으로 AI 에이전트의
다음 행동을 예측하는 것이다. 예측 대상은 파일 읽기, 검색, 수정, 명령 실행,
테스트, 사용자 질의, 계획 수립 등 14개 행동 클래스이며 평가 지표는 Macro-F1이다.

최종 목표는 Public Macro-F1 0.794 이상이었으나, 최종 최고 성능은
**0.7893958473**이었다.

## 2. 데이터와 검증 설계

학습 데이터는 70,000개 샘플로 구성된다. 동일 작업 세션의 여러 step이 존재하므로
행 단위 무작위 분할은 검증 누수를 만들 수 있다. 이를 방지하기 위해 샘플 ID의
`-step_` 앞부분을 그룹으로 사용한 `StratifiedGroupKFold` 5-fold 검증을 적용했다.

검증 시 다음 원칙을 사용했다.

- 동일 세션은 train과 validation에 동시에 포함하지 않음
- OOF 확률은 해당 행을 학습하지 않은 모델에서 생성
- 후처리 파라미터는 가능한 경우 outer-fold 또는 nested 방식으로 평가
- full-data 결과와 cross-fitted 결과를 분리해 보고
- Public leaderboard 결과와 로컬 검증 결과를 별도 관리

## 3. 최종 모델

### 3.1 Qwen 구간별 분류기

Qwen2.5-Coder-0.5B를 backbone으로 사용하고 입력을 다음 구간으로 분리했다.

- current
- action
- history
- 전체 attention 구간

각 구간의 hidden state를 masked mean pooling하고 256차원 projector를 거쳐
결합했다. 여기에 structured feature MLP 출력을 추가해 최종 512차원 표현을
구성했다.

### 3.2 다중 head

- Action head: 14개 최종 행동 클래스
- Family head: 상위 행동 family

최종 logit은 action logit, family logit, class prior 보정을 결합해 계산했다.

### 3.3 OOF teacher distillation

5-fold teacher OOF logit을 soft target으로 활용했다. 학습 손실에는 다음 요소가
포함됐다.

- hard-label final loss
- hard-label action loss
- family loss
- final-logit distillation
- action-logit distillation
- family-logit distillation
- teacher confidence weighted CE

### 3.4 LightGBM 앙상블

Qwen 확률과 handcrafted feature 기반 LightGBM 확률을 log-probability 공간에서
결합했다. 전체 클래스 공통 Tree weight `0.15`가 Public에서 가장 안정적이었다.

## 4. 최종 설정

```json
{
  "architecture": "qwen_distill_v12",
  "base_model": "Qwen2.5-Coder-0.5B",
  "full_max_updates": 3500,
  "head_lr": 0.0004,
  "lora_lr": 0.00002,
  "tree_blend_weight": 0.15,
  "action_temperature": 1.4,
  "family_weight": 1.25,
  "prior_beta": -0.85
}
```

## 5. 주요 실험

### 5.1 Global Tree weight

| Tree weight | Public Macro-F1 |
|---:|---:|
| 0.10 | 0.7890965708 |
| 0.15 | **0.7893958473** |
| 0.20 | 0.7886778450 |

0.15가 최적이었고 주변 weight는 하락했다. 추가 미세 sweep은 기대효과가 낮다고
판단했다.

### 5.2 Teacher 및 soup

- Teacher 50:50 blend: 0.7893696524
- cur160 soup: 0.7893124612
- selective teacher: 0.7887775583

Teacher 정보를 직접 결합해도 최종 cur160을 넘지 못했다.

### 5.3 Gated Tree

Gated Tree V27은 로컬 검증에서 개선 신호가 있었으나 Public 점수는
0.7881191369로 하락했다. teacher OOF와 full student의 오차 분포 차이 때문에
gate가 일반화되지 못한 것으로 해석했다.

### 5.4 Viterbi

세션 전이행렬 기반 Viterbi는 teacher OOF에서 약 +0.0068의 큰 개선을 보였다.
그러나 로컬 sample test 진단 결과는 다음과 같았다.

```text
total_rows=5
sessions=5
multi_step_sessions=0
changed_rows=0
```

실제 추론 입력에서 같은 세션의 여러 target step이 동시에 제공되지 않으면
Viterbi는 적용할 연속 경로가 없다. Public 점수도 baseline과 동일했다.
따라서 OOF의 큰 개선은 배포 입력 구조와 일치하지 않는 검증 방식에서 발생한
것으로 판단하고 폐기했다.

### 5.5 Last-action prior

- Baseline OOF: 0.780297
- Full OOF 최고 gain: +0.000930
- Nested mean gain: +0.000352
- Nested worst fold: -0.000681

전체 OOF 최적값은 양수였지만 nested 일반화 이득이 작아 폐기했다.

### 5.6 Alternate token budget

Fold 0의 Qwen+Tree baseline은 0.785194였다.

- current-heavy 최고: -0.000030
- history-heavy 최고: -0.000441
- action-heavy 최고: -0.000690

대체 view가 기본 모델과 불일치한 행에서 대체 view의 승률은 약 31~33%였다.
기본 입력 구성이 우세하므로 폐기했다.

### 5.7 Classwise Tree weight

- Nested mean gain: +0.000549
- Positive folds: 5/5
- Student fold 0 gain: +0.000183
- Consensus OOF reference: +0.001332
- Full-data reference: +0.001515

방향성은 일관됐지만 실제 student fold에서의 개선 폭이 너무 작았다.
full-data reference는 동일 OOF를 보고 weight를 선택한 낙관적 결과이므로
최종 판단에는 사용하지 않았다.

### 5.8 1.5B 모델

Qwen 1.5B 모델은 로컬 검증 약 0.7745로 0.5B보다 낮았고 제출 패키지 용량도
1GB 제한을 넘기기 쉬웠다. 모델 크기 확대가 성능 향상으로 이어지지 않았다.

## 6. 최종 성능

| 모델 | Public Macro-F1 |
|---|---:|
| V23 Qwen + Tree | 0.7870583827 |
| Rebuilt cur160 | 0.7888347480 |
| Final cur160 | **0.7893958473** |

최종적으로 추가 gate, specialist, transition, classwise weight를 사용하지 않은
원본 cur160 Qwen–Tree 앙상블을 선택했다.

## 7. 실패 원인 분석

1. **Teacher OOF와 full student의 분포 차이**  
   OOF에서 학습된 selector와 gate가 full student 오차에 그대로 적용되지 않았다.

2. **배포 불가능한 검증 정보 사용**  
   Viterbi는 validation 세션의 여러 target 행을 동시에 이용했지만 실제 test
   입력에서는 동일 구조를 사용할 수 없었다.

3. **후처리 과적합**  
   클래스별 weight나 파라미터 grid는 full OOF에서 개선됐지만 nested 또는
   student holdout에서는 이득이 축소됐다.

4. **기본 모델의 강한 저마진 성능**  
   대체 view와 specialist가 기본 모델과 다른 예측을 낸 행에서도 기본 cur160의
   정답률이 더 높았다.

5. **모델 크기보다 입력·검증 설계가 중요**  
   1.5B 확대보다 0.5B의 구간별 pooling, structured features, distillation,
   Tree blend가 더 효과적이었다.

## 8. 결론

Qwen2.5-Coder-0.5B 기반 구간별 분류기와 structured features, OOF teacher
distillation, LightGBM log-probability blend를 결합해 Public Macro-F1
0.7893958473을 달성했다.

목표 점수 0.794에는 도달하지 못했지만 실험 과정에서 모델 성능뿐 아니라
검증 누수, 배포 입력 구조, OOF와 full model 간 분포 차이를 체계적으로
확인했다. 최종 모델은 로컬 최고점보다 Public 재현성과 안정성을 우선해
선정했다.
