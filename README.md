# Exp3: AgentAlign 기반 카테고리별 인과적 refusal-direction 탐색

현재 메인 실험은 `configs/exp3_agentalign_llama31.yaml` / `scripts/run_exp3.py`다.
과거 Exp1/H2 코드는 재현 목적으로 보존한다. H2 설명은 이 문서 아래에 남겨 두었다.

이 재설계는 과거 결과의 실패 양상을 직접 겨냥한다. Exp1은 선택된 11개 중
10개가 자체 feasibility 기준을 통과하지 못했지만 fallback 방향이 결과에
포함되었다. 유해로 취급한 입력의 WildGuard 유해 판정률은 범주별로
약 0.11–0.56이었고 파싱 실패도 있었다. H2는 step별 표현 분리만 보았으며
후반 step의 held-out 쌍은 6–10개 수준이었다. 따라서 어느 결과도 실제
agent에서 안전하게 거절을 유발하는 위치의 증거로 취급하지 않는다.

## 실험에서 실제로 검증하는 것

- 동일한 **Llama 3.1 8B Instruct**가 AgentAlign 유해·정상 과제를 모의 도구와
  함께 처리한다. AgentAlign에 저장된 타 모델의 답변·도구 궤적은 activation이나
  답안으로 사용하지 않는다.
- 공개 원본 중 AgentAlign 자체의 정상 과제만 사용한다. ToolACE/Glaive 유래 정상
  과제 4,010개는 출처 교란을 피하려고 제외한다. 유해 8개 카테고리와 정상 과제를
  **노출된 도구 이름의 정확한 집합**으로 매칭하고, 각 도구 집합 안에서 같은 수로
  뽑는다. 같은 추상 `pattern`은 train/val/test 중 한 곳에만 배정된다.
- 방향 후보: `mean(resid_pre | harmful) - mean(resid_pre | benign)`.
  초기 agent 의사결정 직전의 프롬프트에서 위치 `-5..-1`, 레이어 `0..31`을 모두
  계산한다. 이 수식은 Arditi et al.의 방향 추출 방법과 같다. 사용 모델·도구·데이터
  환경이 달라 최종 벡터가 같다고 가정하지 않는다.
- 주 분석의 선택 가능 레이어는 `0..24`다. 원 연구의 마지막 20% 제외 규칙에 맞춘
  것이다. `25..31`도 점수와 행동을 기록하지만 **탐색적 후기 레이어**로 별도
  표시한다. 23–24번이 앞서면 22–26번 인접 후보를 추가로 검증하고
  `boundary_sensitive=true`를 기록한다. 경계에서 골랐다는 사실을 한 레이어의
  유일한 위치에 대한 증거로 취급하지 않는다.
- 빠른 proxy 점수는 전체 160개 좌표를 훑는 데만 사용한다. 최종 후보 판정은 분리된
  validation 과제에서 agent가 직접 생성한 최대 4단계 도구 사용 궤적으로 한다.
  방향 **제거** 시 유해 요청 거절 감소, 방향 **추가** 시 유해 요청 거절 증가·위험
  행동 감소, 정상 요청 과잉거절·성공률·유효한 모의 도구 사용률 손실 각각
  최대 5%p를 모두 요구한다.
  만족 후보가 없으면 **검증한 후보 중 채택할 방향 없음**이다. 과거 코드의
  fallback은 쓰지 않는다.
- 긴 proxy 스캔 전에 validation baseline으로 데이터·판정기 적합성을 검사한다.
  유해/정상 요청의 독립적 위험 판정, 파싱·불확실률, 정상 과제의 실제 모의 도구
  사용률을 점검한다. 정상 과제 도구 사용률이 50% 미만이면 단순 채팅 실험에
  가까워지므로 해당 카테고리의 선택을 중단한다. `selection/*_judge_quality.json`에
  원인을 남긴다. 이 검사는 agent 능력의 충분조건은 아니며 최소 관문이다.
- 로컬 Qwen2.5-7B-Instruct 판정은 거절/위험 행동/정상 성공의 **추정**이다.
  길이 제한으로 판정 입력을 자르지 않고 판정 불가로 표시하며, 파싱 실패·불확실
  사례도 감추지 않는다. 모든 원문/도구 호출/도구 결과/
  기본·추가·제거 답변을 `manual_review/*.html`에 나란히 남긴다. 사람이 검토하기
  전까지 `human_confirmed=false`다. test의 paired bootstrap 95% CI와 카테고리
  vs global 비교의 Holm 보정도 보고한다. 희소 카테고리의 5%p 비열등성은 CI가
  넓어 결론 유보가 정상적인 결과일 수 있다.
- `posttool_directions/`는 **동일 모델이 실제로 첫 도구를 호출한 후**의 2번째
  의사결정 activation에서 만든 별도 방향 후보와 초기 방향 cosine을 기록한다.
  첫 도구 호출이 유해/정상 양측에 충분하지 않으면 `not_estimable`로 남긴다.
  이 후속-step 방향은 아직 인과적으로 선택하지 않으므로 최종 방어 방향이라고
  부르지 않는다.
- Agent-SafetyBench는 외부 전이 점검에 쓰되, 안전을 위해 공식 환경 Python을
  실행하지 않는다. `misinformation`·`privacy`에 명확히 대응하는 500개 과제의
  **첫 의사결정만** 도구 스키마와 함께 평가한다. 이것은 공식 전체 episode 점수나
  실제 도구 실행 성능이 아니다. 다른 ASB 위험 범주를 임의로 8개 분류에 대응시키지
  않는다.

## GPU 서버 실행 (Docker 불필요)

```bash
cd ~/agent-jailbreaking
git switch main
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -e .
nvidia-smi
# 먼저 agent tool-use와 판정 품질만 빠르게 확인한다.
bash scripts/run_exp3_tmux.sh exp3-preflight configs/exp3_agentalign_llama31.yaml 0 preflight

# preflight_report.json의 9개 항목이 모두 passed=true일 때 본 실험을 시작한다.
bash scripts/run_exp3_tmux.sh exp3-agent-refusal configs/exp3_agentalign_llama31.yaml 0 all
```

마지막 인자 `0`은 **물리 GPU 0번**이다. 다른 GPU를 사용할 때 이 숫자만 바꾼다.
모델 접근 권한이 승인된 Hugging Face 계정으로 서버에서도 로그인되어 있어야 한다.

```bash
bash scripts/tmux_status.sh exp3-agent-refusal
tmux attach -t exp3-agent-refusal
```

연결을 끊어도 tmux 세션은 유지된다. 재실행은 동일한 명령을 새 세션 이름으로
시작하면 체크포인트에서 이어 간다. 설정이나 코드가 달라졌으면 기존 `output_dir`
결과를 섞지 않고 새 디렉터리를 지정해야 한다. 데이터와 공식 모의 도구 코드는
고정 리비전/체크섬으로 검사한다. 모의 도구는 실제 셸·메일·결제·네트워크 서비스에
접속하지 않는다.

주요 결과는 `runs/exp3_agentalign_category_causal_llama31_v5/` 안의
`directions/`, `proxy/`, `selection/`, `test_report.json`, `manual_review/`,
`posttool_directions/`, `external_first_decision/`에 기록된다.
`selection/*.json`의 `selected: null`은 오류가 아니라 데이터·판정기 관문 또는
인과 검증을 통과한 **shortlist 후보**가 없다는 결과다. 전체 좌표에 방향이
존재하지 않는다는 증거는 아니다.

## 구현·해석상의 제한

이것은 GPU 서버에서 최초 실험이 실행되기 전에는 결과를 보증하지 않는다. 정상 과제
성공은 공식 정답 판정기가 없어 로컬 모델의 추정이다. 모의 도구의 임의 출력을
과제·step별로 고정해 조건 간 비교를 짝지었지만, 실제 API의 현실성까지 검증한
것은 아니다. Proxy shortlist 밖의 좌표는 행동 검증을 하지 않으므로 “모든 160개
좌표 중 전역 최적”이라고 해석해서는 안 된다. 리니어 프로브와 런타임 카테고리
라우팅은 이번 실험에 포함되지 않는다.

---

# Historical H2: LLM vs Agent-Step Refusal Directions

현재 메인 실험은 `meta-llama/Llama-3.1-8B-Instruct`에서 일반 chat 입력으로 구한
refusal direction과 AgentLens trajectory의 각 step에서 구한 direction을 비교한다.

이번 실행은 representation 비교만 수행한다. 다음 항목은 포함하지 않는다.

- category별 direction
- AgentHazard
- linear probe와 runtime routing
- activation addition/ablation
- full response generation과 WildGuard
- 실제 tool 실행

기존 category 실험 코드는 이전 결과 재현을 위해 남아 있지만, 현재 실행 진입점은
`scripts/run_h2.py`와 `configs/h2_agent_step_llama31.yaml`이다.

## 연구 질문

1. 일반 LLM refusal direction과 전체 agent direction은 같은가?
2. Agent trajectory의 step별 direction은 LLM direction과 얼마나 유사한가?
3. Step이 진행되면서 direction의 cosine, norm, held-out 분리 성능이 변하는가?
4. LLM direction과 agent-step direction 중 어느 쪽이 held-out agent state를 더 잘
   분리하는가?

이번 결과는 인과적으로 검증된 최종 refusal direction이 아니라
`candidate agent-step refusal directions`다. 안정적인 차이가 관찰된 뒤 별도
addition/ablation 실험으로 인과성을 검증한다.

## 고정 좌표

Arditi et al.이 Llama-3-8B-Instruct에서 보고한 좌표를 외부 기준으로 고정한다.

```text
layer = 12 (0-based)
position = -5 = <|eot_id|>
```

모든 LLM 및 agent direction의 primary 비교는 같은 layer와 position에서 수행한다.
추가로 position `-5`에서 32개 layer 전체의 same-layer cosine을 기록하여 agent에서
표현 위치가 이동하는지 탐색한다. Agent 결과를 보고 primary 좌표를 다시 고르지 않는다.

## 데이터와 direction 정의

### LLM reference

Arditi et al. 공식 저장소의 기존 split을 그대로 내려받는다.

```text
data/raw/refusal_direction/dataset/splits/
  harmful_train.json
  harmful_test.json
  harmless_train.json
  harmless_test.json
```

```text
r_llm = mean(resid_pre | harmful chat)
      - mean(resid_pre | harmless chat)
```

Train은 side당 최대 128개, test는 side당 최대 64개를 seed 42로 고정 추출한다.

### Agent direction

AgentLens 공식 `MAS/LLaMA/train.json`, `test.json`만 사용한다.

- `label=1`: harmful execution state
- `label=0`: benign/safety-aware state
- normalized task description 단위로 70/30 train/test를 다시 분리하여 동일 task가
  서로 다른 split에 들어가지 않게 한다. H2에는 validation-time 선택이 없으므로 별도
  validation split을 만들지 않는다.
- 각 step 안에서 label 1과 label 0을 같은 수로 뽑는다.

```text
r_agent_step_s = mean(resid_pre | label=1, step=s)
               - mean(resid_pre | label=0, step=s)
```

`agent_all`은 eligible step별 balanced sample을 합쳐 계산한다. 따라서 데이터가 많은
특정 step이나 긴 trajectory가 전체 direction을 지배하지 않는다.

현재 AgentLens 표본 분포에서는 train label별 최소 16개, test label별 최소 6개를
동시에 만족하는 step 1-5가 분석 대상이다. Step 6 이후는 label 0 trajectory가 너무
적으므로 방향이 없다는 결론을 내리지 않고 `insufficient support`로 제외한다.

## 산출 지표

- LLM, Agent-all, 각 Agent-step direction의 primary cosine matrix
- position `-5`에서 32개 layer 전체의 pairwise cosine과 norm
- held-out projection AUROC, Cohen's d, harmful/benign mean difference
- LLM direction을 agent step에 투영한 cross-domain 성능
- agent-step direction을 LLM 및 다른 step에 투영한 성능
- train direction bootstrap cosine 95% interval, 500회

Linear probe를 학습하지 않는다. Projection AUROC는 학습된 분류기가 아니라 고정된
한 direction의 held-out 분리 성능이다.

## 서버 설치 및 데이터 준비

```bash
cd ~/agent-jailbreaking
git switch main
git pull --ff-only origin main

source .venv/bin/activate
python -m pip install -e ".[dev]"

# 공식 Arditi split 4개를 다운로드하고 JSON 구조를 검증한다.
python -u scripts/prepare_h2_data.py

# AgentLens가 이미 없다면 한 번만 실행한다.
git clone https://github.com/EddyLuo1232/AgentLens.git data/raw/AgentLens
```

Llama 3.1 gated access와 Hugging Face 로그인이 필요하다.

```bash
huggingface-cli login
python -u scripts/check_gpu_env.py
```

## tmux 실행

물리 GPU 0을 사용하는 예시는 다음과 같다.

```bash
bash scripts/run_h2_tmux.sh \
  h2-agent-step \
  configs/h2_agent_step_llama31.yaml \
  0
```

상태 확인:

```bash
bash scripts/tmux_status.sh h2-agent-step
```

실시간 확인:

```bash
watch -n 10 'bash scripts/tmux_status.sh h2-agent-step'
```

`watch`를 종료하려면 `Ctrl-C`를 누른다. tmux에 직접 접속하려면:

```bash
tmux attach -t h2-agent-step
```

분리는 `Ctrl-b d`다. SSH 또는 VS Code를 종료해도 tmux 내부 실험은 계속된다.

## Stage와 재개

```text
prepare -> extract -> analyze
```

- `prepare`: 데이터 개수와 step별 label 균형을 검사하고 eligible step을 확정한다.
- `extract`: LLM/AgentLens activation을 GPU에서 추출해 checkpoint로 저장한다.
- `analyze`: CPU에서 direction, cosine, projection, bootstrap을 계산한다.

개별 실행:

```bash
python -u scripts/run_h2.py --config configs/h2_agent_step_llama31.yaml --stage prepare
python -u scripts/run_h2.py --config configs/h2_agent_step_llama31.yaml --stage extract
python -u scripts/run_h2.py --config configs/h2_agent_step_llama31.yaml --stage analyze
```

Activation cache가 존재하면 extract stage는 해당 항목을 건너뛴다. Config의 model,
position, 데이터 또는 split 조건을 바꿀 때는 새 `output_dir`을 사용한다.

## 출력

```text
runs/h2_llm_vs_agent_step_refusal_llama31_v1/
  resolved_config.json
  data_summary.json
  position_label.json
  cache/
    llm_train_harmful.pt
    llm_train_harmless.pt
    llm_test_harmful.pt
    llm_test_harmless.pt
    agent_train.pt
    agent_train_metadata.json
    agent_test.pt
    agent_test_metadata.json
  directions/
    step_directions.pt
  analysis/
    layerwise_cosine.csv
    primary_cosine.csv
    projection_metrics.csv
    bootstrap_cosine.csv
    summary.json
```

로그는 batch마다 JSON Lines로 출력되며 완료량, 백분율, 경과 시간, ETA, GPU memory와
checkpoint 경로를 포함한다.

## 예상 시간

RTX PRO 6000 Blackwell 96GB 한 장, batch size 8, 모델이 이미 Hugging Face cache에
있는 조건의 보수적인 예상이다.

```text
데이터 준비/검증:       1-3분
모델 로드:              1-3분
LLM activation 추출:    2-6분
AgentLens activation:  10-25분
CPU 분석/bootstrap:     2-8분
합계:                  약 20-45분
```

AgentLens context가 대부분 4096 token 부근이거나 모델을 처음 다운로드하는 경우
45-75분까지 늘어날 수 있다. WildGuard와 autoregressive 128-token 생성이 없으므로
이전 category 실험처럼 수 시간 이상 걸리는 구조는 아니다.

## 검증

```bash
pytest
ruff check .
```

## 참고자료

- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Official refusal-direction implementation](https://github.com/andyrdt/refusal_direction)
- [Applying Refusal-Vector Ablation to Llama 3.1 70B Agents](https://arxiv.org/abs/2410.10871)
- [AgentLens repository](https://github.com/EddyLuo1232/AgentLens)
