# 실험 로그

## 기준선과 최종 모델

| 구분 | 핵심 변경 | 결과 | 판단 |
|---|---|---:|---|
| V4 | Qwen 구간별 pooling + structured features | 기반 모델 | 유지 |
| V12 | OOF teacher distillation | fold 0 약 0.783 | 유지 |
| V23 | Qwen + LightGBM blend | Public 0.7870583827 | 이전 최고 |
| cur160 | 학습 및 후처리 조정, Tree 0.15 | Public 0.7893958473 | 최종 최고 |

## 모델 계열

| 실험 | 결과 | 판단 |
|---|---:|---|
| 1.5B Qwen | validation 약 0.7745 | 폐기 |
| V27 Tree-teacher gated | validation 0.783466, Public 0.7881191369 | 폐기 |
| Hard-negative / specialist 계열 | 최고 Public 미개선 | 폐기 |
| V13/V14 구조 변경 | cur160 미개선 | 보조 기록 |

## 앙상블 및 후처리

| 실험 | 결과 | 판단 |
|---|---:|---|
| Tree weight 0.10 | Public 0.7890965708 | 하락 |
| Tree weight 0.15 | Public 0.7893958473 | 최종 |
| Tree weight 0.20 | Public 0.7886778450 | 하락 |
| Teacher 50:50 | Public 0.7893696524 | 거의 동점 |
| cur160 soup | Public 0.7893124612 | 하락 |
| Selective teacher | Public 0.7887775583 | 하락 |
| Postprocess candidate | Public 0.7888850680 | 하락 |

## 배포 구조 검증

| 실험 | 로컬 결과 | 배포/제출 결과 | 판단 |
|---|---:|---:|---|
| Viterbi transition | OOF 약 +0.0068 | sample test 변경 0행, Public 동점 | 폐기 |
| Last-action prior | nested +0.000352 | 미제출 | 폐기 |
| Alternate token budget | 최고 -0.000030 | 미제출 | 폐기 |
| Classwise Tree | nested +0.000549, student +0.000183 | 미제출 | 폐기 |

## 최종 결정

추가 미세조정은 대부분 +0.001 미만의 약한 신호이거나 student holdout과
Public에서 재현되지 않았다. 따라서 정확히 재현된 원본 cur160 제출물을 최종본으로
선정했다.
