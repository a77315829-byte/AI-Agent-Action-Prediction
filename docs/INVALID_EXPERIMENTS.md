# 유효하지 않거나 해석에 주의가 필요한 실험

## 1. Fold 1–4 student validation leakage

초기에는 fold 0에서 사용한 `qwen_segment_v4` 초기 체크포인트를 유지한 채
`--eval-fold`만 변경해 fold 1–4 distillation 평가를 수행했다.

하지만 해당 V4 체크포인트는 fold 0만 제외하고 학습됐기 때문에 fold 1–4의
validation 행을 이미 학습 과정에서 본 상태였다. 따라서 fold 1–4 결과는
독립 검증이 아니다.

특히 fold 4 초기 Macro-F1 약 0.8471은 실제 일반화 성능 향상이 아니라
초기 체크포인트의 validation 노출로 해석했다.

다음 결과는 최종 비교에서 제외한다.

- `distill_cur160_eval_fold1`
- `distill_cur160_eval_fold2`
- `distill_cur160_eval_fold3`
- `distill_cur160_eval_fold4`

유효한 student holdout은 학습에서 제외된 fold 0 결과만 사용했다.

## 2. Viterbi OOF와 실제 test 구조의 불일치

OOF screen은 같은 세션의 여러 target 행을 묶어 전체 경로를 결정했다.
그러나 로컬 sample test는 5개 행이 모두 서로 다른 세션이었다.

```text
total_rows=5
sessions=5
multi_step_sessions=0
changed_rows=0
```

따라서 OOF의 큰 개선은 추론 시 이용할 수 없는 미래 target 행 또는 동시 target
행을 사용한 결과다. 모델 학습 누수라기보다 배포 정보 가용성 불일치다.

## 3. Full-data classwise weight

전체 OOF를 보고 클래스별 weight를 선택한 뒤 동일 OOF에 평가한
`full-data reference`와 `consensus reference`는 낙관적이다.

최종 판단에는 outer-fold cross-fitted gain과 독립 student fold 0 gain만 사용했다.

## 4. Public leaderboard 해석

Public 점수 차이가 작은 실험은 공개 평가 샘플 구성에 따라 순위가 바뀔 수 있다.
따라서 다음을 구분해 기록했다.

- 검증 파라미터 선택용 OOF
- 독립 student holdout
- Public leaderboard
- 실제 배포 가능한 입력 정보

최종 모델 선택은 단일 최고 로컬 점수보다 재현성과 제출 안정성을 우선했다.
