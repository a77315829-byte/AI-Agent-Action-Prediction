# cur160 teacher 공유 (LB 0.7893958, 이전 best 0.7870907 대비 +0.00231)

## 한 줄 요약
6단계 파이프라인 **구조는 그대로**, distill 2단계의 **OOF teacher만** `current` 토큰 예산을
**96 → 160**으로 키워서 다시 만든 것. teacher가 긴 자연어 trio 요청(ask/plan/web)을
96 토큰에서 잘라먹던 걸 160으로 살렸더니 distill이 더 나은 soft target을 배움.

## 성능
- teacher 자체(OOF overall macro): 표준 0.778213 → **cur160 0.778976** (fold별: 0.7787/0.7803/0.7767/0.7773/0.7810)
- 이 teacher로 distill(mode=full, --max-updates 3500) → tree(f432631a) blend 0.15 → **LB 0.7893958473**
- 검증: 이전 best 제출(blend015)과 zip 비교 시 **신경망 엔진 파일(heads/metadata/qwen_int8)만 상이**,
  나머지(tree/postprocess/blend weight/script) 전부 byte-identical → 순수 teacher 효과.

## 폴더 구성
- `code/run_oof_clean.py` — OOF teacher 빌더 (원본 run_qwen_v4_oof_folds.py의 수정본)
- `code/run_oof_clean_vs_original.diff` — 원본 대비 diff (바뀐 부분 전체)
- `code/commands.txt` — teacher 생성 → distill → 패키징 정확한 커맨드
- `teacher_cur160/oof_logits_all_70000.npz` — ★ 이게 teacher 본체 (distill이 소비하는 로짓)
- `teacher_cur160/oof_summary.json`, `run_manifest.json`, `oof_classification_report.txt` — provenance/설정
- `teacher_cur160/per_fold_configs/` — 폴드별 metadata/postprocess/status (설정만, 가중치 제외)

## 원본에서 바뀐 코드 (run_oof_clean.py, 원본=run_qwen_v4_oof_folds.py)
1. 세그먼트 예산 인자를 v4 globals에 연결 (`--max-length / --current-budget / --history/action/meta-budget`)
   → cur160은 `--max-length 320 --current-budget 160`로 실행.
2. fold 0을 재사용(정우 fold0-logits)하지 않고 **fresh 학습** 허용 (split_seed=42 정렬, val=14284).
3. fold-0 fresh일 때 validate_fold0 스킵.
4. output-dir 이미 있으면 abort하는 가드.
5. (오늘 추가, cur160엔 미사용) `--browse-smooth` 플래그: browse-true 행(클래스<4)에만 per-sample
   label smoothing. 기본값 0.0이라 **cur160 teacher는 영향 없음** (train_engine_exp.py에서 포팅).

## 주의
- teacher는 정우 per-fold-best 체크포인트 정책 그대로 유지 → "완전 정직 OOF"가 아니라
  **표준 teacher와 동일 정책 하의 상대 비교**. threshold/방향은 전이되나 정밀 전이는 제출로 확정.
- 현재 진행 중: cur160 + browse-smooth 0.1 (± family-weight 0.5) 변종 2개 (아직 미완료).
