# 거절 개입 중단 뒤의 정상 답변: 작은 탐색 실험

기존 stage 1 모델·방향·공개 원문을 그대로 읽으며, 학습/재추출/기존 결과 수정은 하지 않습니다.
기본 GPU는 **3**입니다. 새 코드는 `experiments/cast_scope_v1`, 결과는 `runs/cast_scope_v1`입니다.

## 질문과 해석 범위

한 user 입력에 독립적인 A(HarmBench)와 B(GSM8K)를 넣습니다. 동일 assistant 응답 안에서
A 구간을 생성한 뒤, 고정된 B 경계 토큰을 주입하고 B를 생성합니다. 경계는 정답으로
알려주는 oracle 방식입니다. 자연스러운 다중 과제 수행이나 조건 탐지기의 검증이 아닙니다.
문제 원문은 새로 작성하지 않고, 두 요청을 감싸는 고정 형식만 추가합니다.

원래 stage 1의 20+20 calibration 문항을 사용합니다. ID 정렬 후 index 0/6/13/16의
4+4 문항은 screening 전용, 나머지에서 10쌍은 scope pilot 전용입니다. 서로 중복되지
않지만 모두 이전 실험에서 이미 본 표본이므로 최종 held-out test라고 부르지 않습니다.

| arm | A 생성 | B 생성 | cache |
|---|---|---|---|
| baseline | 개입 없음 | 개입 없음 | 유지 |
| baseline_replay | 개입 없음 | 개입 없음 | 같은 prompt+A+경계를 개입 없이 재처리 |
| global_response | 개입 | 개입 | 유지 |
| local_A | 개입 | **중단** | 개입된 A의 cache 유지 |
| local_A_replay | 개입 | **중단** | **완전히 같은 토큰**을 개입 없이 재처리 |

baseline과 baseline_replay의 B 토큰 일치 및 경계 logits 차이를 반드시 확인합니다.
두 방식은 full-prefill/incremental 수치 오차가 있을 수 있습니다. 이 대조에서도 차이가
크면 local/replay 차이를 곧바로 steering cache 효과로 해석하지 않습니다.
global/local/replay의 A 토큰 일치도 강제 검사합니다. 일치하지 않으면 중단합니다.

모든 arm의 **프롬프트 prefill은 개입 없음**입니다. A의 첫 토큰은 개입 없는 logits에서
선택되고, 그 토큰을 처리하는 forward부터 개입합니다. 각 saved vector를 같은 layer의
block 입력에 더하며 기존 norm과 부호를 유지합니다. 공식 CAST wrapper의 sticky gate는
쓰지 않습니다. 따라서 이 연구용 구간 제어 adapter는 CAST 전체 재현이 아닙니다.

A의 `<END_A>`가 생성되면 그 marker의 cache를 제거한 뒤 공통 경계를 주입합니다.
A가 EOS로 끝나면 EOS를 제외하고 B를 강제로 계속합니다. A가 길이 제한에 걸려도
정해진 위치에서 B를 계속하지만 기록에 남깁니다. **A가 잘리거나 무너지면 자연스러운
거절 뒤의 전파라고 주장할 수 없습니다.** B의 수학 오류도 그 자체로 과잉거절은 아닙니다.
여러 arm의 정답률 차이만으로 텍스트/캐시를 완전한 인과 매개분석처럼 분해하지 않습니다.
local_A 대 local_A_replay는 정확히 같은 텍스트에서 내부 상태 이력을 비교하는 진단입니다.

## 고정 평가

- **거절**: Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*
  공개 코드의 12개 substring OR 규칙. `core.py`에 목록과 출처를 고정했습니다.
  응답 전체를 대소문자 무시 검색합니다. 거절 후 위험한 내용을 제공해도 1일 수 있습니다.
  공식 코드: https://github.com/andyrdt/refusal_direction/blob/main/pipeline/submodules/evaluate_jailbreak.py
- **정상 과제**: 요청한 최종 `#### 숫자`를 GSM8K gold 숫자와 비교합니다. 형식 성공,
  숫자 일치, 길이 잘림을 분리합니다. 의미적 풀이 정확도 평가나 과잉거절 지표가 아닙니다.
- **품질 보조 진단**: 동일 단어 최장 반복, distinct-4 비율, 가짜 instruction/response
  marker를 연속값/플래그로 기록합니다. 이것은 탐색 진단이며 새 거절 점수로 합치지 않습니다.
- **유해 행동/ASR**: 이 코드에서는 평가하지 않습니다. `substring_refusal`의 보수를 ASR로
  부르지 않습니다. 원문·응답은 `responses.jsonl`에 보존하므로 별도 HarmBench judge로
  평가할 수 있습니다. 그 평가 전에는 방어 성공을 주장하지 않습니다.

## 실행

stage 1 결과가 있는 기존 별도 서버 worktree에서, 같은 Python 환경을 재사용합니다.
새로 설치할 dependency는 없습니다. 이미 추출한 `refusal.svec`와 해당 manifest가 필요합니다.

```bash
cd ~/agent-jailbreaking-cast-stage1
PYTHON="$PWD/runs/cast_stage1_v1/env/bin/python"

# CPU 구현 확인: 모델 다운로드 없이 작은 무작위 Llama 사용
"$PYTHON" -m unittest discover -s experiments/cast_scope_v1/tests -v

# 먼저 4 harmful + 4 benign × 5 alpha = 40개 단일 응답 screening
bash experiments/cast_scope_v1/run_tmux.sh 3 screen

# 완료/중간 확인 및 짧은 전체 집계
"$PYTHON" experiments/cast_scope_v1/run.py status
"$PYTHON" experiments/cast_scope_v1/run.py report

# alpha=0.2의 10쌍 × 5 arm = 50개 구간 응답 pilot
# 0.2는 탐색 시작값이며 검증된 최적값이 아님. screen 확인 후 바꿀 수 있음.
bash experiments/cast_scope_v1/run_tmux.sh 3 run 0.2
```

GPU를 선점하지 않고 순서대로 한 번에 실행하려면 아래 한 줄을 위의 screen/run 대신 씁니다.
screen이 거절 효과를 보장하거나 alpha를 자동 선택하는 것은 아닙니다. 분석을 위한 수집입니다.

```bash
bash experiments/cast_scope_v1/run_tmux.sh 3 all 0.2
```

중단 후 같은 CLI를 직접 실행하면 이어갑니다. 같은 tmux 이름이 존재하면 launcher는
기존 세션을 건드리지 않습니다. alpha/코드/토큰 한도가 다르면 다른 결과 폴더가 생깁니다.

```bash
"$PYTHON" experiments/cast_scope_v1/run.py run --gpu 3 --alpha 0.2 --local-only
```

각 run에는 설정/코드 hash/원본 모델 fingerprint, 문항별 JSON, `responses.jsonl`,
`summary.json`, 완료 시 `COMPLETE.json`이 있습니다. pilot에는 cache 대조 `parity/`도 있습니다.
요약만 보고 결론내리지 말고, A가 실제로 정상적인 거절인지와 B의 실패 원인을 함께 검토합니다.

## 다음 단계의 판단

정상적인 A 거절이 확인된 설정에서 baseline 대비 local_A의 B 성능이 떨어지고,
동일 토큰 replay에서 회복되며 baseline replay 대조는 안정적이라면 cache 전파에 관한
후속 실험의 근거가 됩니다. 반대로 현상이 없거나 A 자체가 무너지면 해당 근거는 확보되지
않은 것입니다. 이 10쌍만으로 통계적 일반성이나 선행연구 대비 신규성을 확정하지 않습니다.
