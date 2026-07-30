# 공개 저장소 선별 기준

## 포함

- 최종 모델의 핵심 학습·추론 코드
- 모델 구조 변화와 주요 ablation 코드
- Tree, teacher, specialist, Viterbi 실험 코드
- 집계된 confusion matrix 및 클래스별 지표
- 공개 가능한 설정 JSON
- 주요 점수와 실패 분석 문서

## 제외

- 대회 원본 데이터
- 행 단위 오류 샘플 및 prompt 원문
- 모델 체크포인트와 LoRA adapter
- OOF logits, Tree probability, embeddings
- 제출 ZIP 및 임시 빌드 폴더
- 가상환경과 tokenizer 대용량 파일
- 동일 기능의 중복 제출 패키지
- 누수로 판정된 fold 1–4 모델 산출물

`prepare_github_repo.ps1`은 원본 프로젝트를 수정하지 않고 선택된 파일만
별도 저장소 폴더로 복사한다. 누락된 파일은 `results/missing_source_files.csv`에
기록된다.
