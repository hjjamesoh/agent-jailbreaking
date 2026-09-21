# Experiment 1: Category-Specific Refusal Directions in LLM Agents

이 저장소는 첫 번째 메인 실험만 구현한다. 목표는
`meta-llama/Llama-3.1-8B-Instruct`의 multi-turn agent trajectory에서 위험
category마다 서로 다른 refusal direction이 존재하는지 확인하고, 각 category의
최적 `position × layer × alpha`를 선택하는 것이다.

Linear probe 기반 위험 감지와 category routing은 현재 실험 범위가 아니다. 그것은
실험 1의 direction bank가 유효하다는 결과를 확인한 뒤 진행할 실험 2다.

## 실험 질문

1. Agent state에서도 category별 mean-difference refusal direction을 찾을 수 있는가?
2. 가장 좋은 direction의 layer와 post-instruction position은 category마다 다른가?
3. Category direction이 global direction보다 해당 category에서 더 효과적인가?
4. 위험 응답을 줄이면서 benign next-action 성능과 과잉 거절을 보존할 수 있는가?
5. 같은 direction의 효과가 agent step에 따라 어떻게 달라지는가?

이번 실험은 step마다 별도 direction을 만들지 않는다. 모든 step을 direction 추출에
사용하고, 선택 및 held-out 평가 결과를 step별로도 집계한다. Layer와 position은
처음부터 독립 탐색한다.

## 모델과 데이터

### Target model

- `meta-llama/Llama-3.1-8B-Instruct`
- BF16, 비양자화
- 32 transformer layers
- Hugging Face revision `0e9e39f249a16976918f6564b8830bc894c89659`
- greedy decoding

Instruct checkpoint를 사용하는 이유는 연구 대상이 정렬된 agent의 refusal
behavior이기 때문이다. Base model은 chat template와 baseline refusal 자체가 달라
같은 인과 질문을 검증하지 못한다.

### AgentLens MAS/LLaMA

공식 `MAS/LLaMA/train.json`, `test.json`을 사용한다.

- `label=0`: benign 또는 safety-aware step
- `label=1`: harmful execution step
- 본 실험에서는 non-refusal인 `label=0` state를 direction의 공통 safe reference와
  benign utility 평가에 사용한다.
- 동일 task의 여러 step이 train/validation/test에 섞이지 않도록 normalized
  `task_description` 단위로 다시 group split한다.

### AgentHazard

공식 dataset과 공개 execution trajectory를 사용하며 다음 10개 category를 그대로
유지한다.

- `System_RCE`
- `Data_Exfiltration`
- `Persistence_Establishment`
- `Supply_Chain_Poisoning`
- `Destructive_Action`
- `Internal_Reconnaissance`
- `Privilege_Escalation`
- `Resource_Exhaustion`
- `Prompt_Intel_Theft`
- `Compliance_Bypass`

공식 repository의 `traces/*/*.zip`을 압축 해제하지 않고 직접 읽는다. 이미 풀어 둔
`*.jsonl`도 함께 지원한다. 동일 `original_id`의 attack variant와 multi-turn state는
항상 같은 split에 배정된다.

## Direction 탐색과 선택

### 1. 후보 추출

Arditi et al.의 공개 구현과 같은 difference-of-means 정의를 사용한다.

```text
d(category, position, layer)
  = mean(resid_pre | refused harmful category states)
  - mean(resid_pre | non-refusal safe agent states)
```

Llama 3.1 8B의 모든 32개 layer와 assistant generation 직전의 5개
post-instruction position `[-5, -4, -3, -2, -1]`을 탐색하므로 category마다
후보는 160개다. 10개 category direction과 global harmful direction을 별도로 만든다.

### 2. Arditi proxy로 shortlist 생성

각 160개 후보에 대해 다음을 계산한다.

- harmful validation state에서 all-layer directional ablation 후 refusal log-odds
- safe validation state에서 source layer activation addition 후 refusal log-odds
- safe validation next-token distribution의 forward KL
- 마지막 20% layer 제외
- `KL <= 0.1`, safe steering refusal score `>= 0`

필터를 통과한 후보 중 ablation, steering, KL 관점의 상위 후보를 round-robin으로
최대 8개 shortlist에 넣는다. 이 단계의 계산은 선행연구와 비교 가능한 값이며,
최종 선택 기준은 아니다.

### 3. Agent behavior 기준 최종 선택

Shortlist의 각 direction을 L2 unit vector로 정규화하고
`alpha ∈ {0.5, 1, 2, 4, 8}`을 탐색한다. 각 조합으로 전체 next action을 생성한 뒤
`allenai/wildguard` revision
`cbba4823f3e8020e5a74a5e29bf85072def6f2ff`가 다음을 판정한다.

- harmful request 여부
- response refusal 여부
- harmful response 여부

Benign 성능 보존은 두 지표로 제한한다.

- safe-state refusal 증가 `<= 0.05`
- 기록된 benign next action에 대한 reference NLL 증가 `<= 0.20` nat/token

동시에 baseline보다 harmful-response rate나 harmful-state refusal rate가 나빠지는
후보는 feasible 후보에서 제외한다.

제약을 만족하는 조합 중 harmful-response rate가 가장 낮고 refusal rate가 높은
`position × layer × alpha`를 고른다. 제약을 만족하는 후보가 하나도 없으면 최소
constraint violation 후보를 선택하고 artifact에 `constraint_fallback=true`를 남긴다.

### 4. Held-out 평가

Probe 없이 정답 category 라벨만 사용한다.

- pairwise direction cosine matrix
- 모든 source direction × target category의 refusal-log-odds addition/ablation matrix
- same-norm random-direction control
- category direction과 global direction의 held-out full next-action 비교
- safe test state의 refusal rate와 reference NLL
- agent step별 harmful/refusal rate

이 평가는 저장된 trajectory를 counterfactual replay할 뿐 shell command나 tool call을
실제로 실행하지 않는다.

## 이번 코드에 포함되지 않는 것

- harm linear probe 학습 또는 위험 예측
- category linear probe 학습 또는 category 예측
- probe prediction에 따른 runtime direction routing
- 실제 sandbox tool execution

위 항목은 실험 2의 범위다.

## 저장소 구조

```text
configs/main_agent_llama31.yaml
scripts/
  check_gpu_env.py
  prepare_agent_data.py
  run_main.py
  run_all.sh
  run_tmux.sh
  tmux_status.sh
src/task_refusal/
  behavior.py
  config.py
  data.py
  directions.py
  evaluation.py
  hooks.py
  modeling.py
  pipeline.py
  progress.py
  selection.py
tests/
```

## GPU 서버 설치

권장 환경은 96GB VRAM의 NVIDIA RTX PRO 6000 Blackwell이다. 먼저 서버 driver와
CUDA에 맞는 PyTorch build를 설치하고, 그다음 프로젝트 의존성을 설치한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 서버 CUDA/driver에 맞는 PyTorch를 먼저 설치
python -m pip install torch
python -m pip install -e ".[dev]"

huggingface-cli login
python -u scripts/check_gpu_env.py
```

Llama 3.1과 WildGuard는 gated model이므로 Hugging Face 계정에서 두 model의
access 조건을 각각 승인해야 한다.
Target model과 WildGuard 7B를 동시에 BF16으로 올리므로 실행 전 free VRAM을
확인한다.

## 공개 데이터 준비

```bash
mkdir -p data/raw
git clone https://github.com/EddyLuo1232/AgentLens.git data/raw/AgentLens
git clone https://github.com/Yunhao-Feng/AgentHazard.git data/raw/AgentHazard
```

AgentHazard ZIP은 풀 필요가 없다. Manifest만 먼저 검증하려면 다음을 실행한다.

```bash
python -u scripts/prepare_agent_data.py \
  --config configs/main_agent_llama31.yaml
```

누락 category나 group leakage는 model을 올리기 전에 명확한 오류로 중단한다.
Baseline scoring 뒤 refusal-positive 표본 수가 설정값보다 적은 경우에도 자동으로
축소하지 않고 해당 stage를 중단한다.

## tmux에서 메인 실험 실행

먼저 `nvidia-smi` 로 0~5번 물리 GPU 중 free memory가 크고
utilization이 낮은 하나를 고른다. 예를 들어 3번 GPU를 쓰려면 실행
명령의 세 번째 인자로 `3`을 준다.

```bash
nvidia-smi
bash scripts/run_tmux.sh exp1-refusal configs/main_agent_llama31.yaml 3
tmux attach -t exp1-refusal
```

스크립트가 tmux 세션 내부의 `CUDA_VISIBLE_DEVICES=3`을 직접
설정한다. 따라서 실험 프로세스에서는 서버의 물리 GPU 3번이
`cuda:0`으로 보이는 것이 정상이다. 실험 중에는 해당 GPU를 다른 작업과
공유하지 않는다.

분리하려면 `Ctrl-b d`를 누른다. 다른 shell에서 진행 상태를 볼 수 있다.

```bash
bash scripts/tmux_status.sh exp1-refusal
```

로그는 JSON Lines 형식이며 stage, category, position, layer, alpha, 처리량, 경과
시간, ETA, GPU memory와 checkpoint 경로를 출력한다. 전체 action 생성과 WildGuard
판정도 batch마다 진행률을 출력한다.

Stage 순서는 다음과 같다.

```text
prepare -> filter -> extract -> select -> evaluate
```

개별 stage 또는 일부 category만 재개할 수도 있다.

```bash
python -u scripts/run_main.py \
  --config configs/main_agent_llama31.yaml \
  --stage extract \
  --categories System_RCE Data_Exfiltration

python -u scripts/run_main.py \
  --config configs/main_agent_llama31.yaml \
  --stage select \
  --categories System_RCE Data_Exfiltration
```

`evaluate`는 global과 10개 category의 선택이 모두 끝난 뒤 실행한다.

## Checkpoint와 출력

이번 실험은 이전 실험과 섞이지 않도록 다음 새 경로를 쓴다.

```text
runs/exp1_agent_category_refusal_llama31_v1/
```

주요 artifact는 다음과 같다.

```text
data/direction_splits.jsonl
directions/<category>/candidates.pt
directions/<category>/candidate_evaluations.jsonl
directions/<category>/proxy_shortlist.json
directions/<category>/behavior_candidate_evaluations.jsonl
directions/<category>/behavior_generations/*.jsonl
directions/<category>/selected_direction.pt
directions/<category>/selected_direction.json
analysis/cosine_similarity.csv
analysis/addition_delta.csv
analysis/ablation_delta.csv
analysis/random_direction_controls.jsonl
analysis/agent_behavior/heldout_behavior_summary.jsonl
analysis/agent_behavior/heldout_generations/*.jsonl
```

각 category의 후보·alpha 평가가 JSONL에 즉시 append되므로 tmux나 SSH 연결이
끊겨도 같은 명령으로 재개한다. 이미 완료된 항목은 skip한다. Config나 데이터
조건을 바꿀 때는 기존 checkpoint를 재사용하지 말고 새 `output_dir`을 사용한다.

## 기존 GPU checkout 업데이트

먼저 서버의 기존 변경 여부를 확인한다.

```bash
cd /path/to/agent-jailbreaking
git status
```

깨끗하면 다음만 실행한다.

```bash
git switch main
git pull --ff-only origin main
```

서버에 추적 중이거나 새로 만든 코드 변경이 있으면 덮어쓰지 말고 먼저 보관한다.

```bash
git stash push -u -m "gpu-server-before-exp1"
git switch main
git pull --ff-only origin main
```

`data/`와 `runs/`는 Git에서 제외되므로 기존 raw data와 실행 결과는 pull로
삭제되지 않는다. 이번 config는 새 output directory를 사용하므로 과거 checkpoint와
충돌하지 않는다.

## 참고자료

- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Official refusal-direction implementation](https://github.com/andyrdt/refusal_direction)
- [There Is More to Refusal in Large Language Models than a Single Direction](https://arxiv.org/abs/2602.02132)
- [AgentLens paper](https://arxiv.org/abs/2606.22673)
- [AgentLens repository](https://github.com/EddyLuo1232/AgentLens)
- [AgentHazard repository](https://github.com/Yunhao-Feng/AgentHazard)
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [WildGuard](https://huggingface.co/allenai/wildguard)
