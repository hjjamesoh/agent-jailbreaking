# H2: LLM vs Agent-Step Refusal Directions

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
- normalized task description 단위로 train/validation/test를 다시 분리하여 동일 task가
  서로 다른 split에 들어가지 않게 한다.
- 각 step 안에서 label 1과 label 0을 같은 수로 뽑는다.

```text
r_agent_step_s = mean(resid_pre | label=1, step=s)
               - mean(resid_pre | label=0, step=s)
```

`agent_all`은 eligible step별 balanced sample을 합쳐 계산한다. 따라서 데이터가 많은
특정 step이나 긴 trajectory가 전체 direction을 지배하지 않는다.

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
