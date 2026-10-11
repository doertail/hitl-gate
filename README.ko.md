# hitl-gate

**사람이 늘 승인하는 일로 사람을 멈춰 세우지 마세요.** LangGraph human-in-the-loop 단계를 위한, 학습 가능한 "사람에게 물어볼까?" 게이트.

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

[English README](README.md)

![사람 검토 횟수와 놓친 오류의 트레이드오프](docs/tradeoff.png)

함께 제공되는 시뮬레이션 문서 파이프라인([`examples/document_pipeline`](examples/document_pipeline/run.py), 홀드아웃 1,000회 실행. 모든 수치는 시뮬레이션 결과):

- **AlwaysAsk**는 실행당 3.00건을 사람에게 보내고, 100회 실행당 0.9건의 오류를 통과시킵니다.
- 로그 500회로 학습한 **RulePolicy**는 실행당 1.29회만 묻고(**검토 57% 감소**) 100회당 2.2건의 오류를 통과시키며, 총비용은 실행당 3.23에서 1.84로 줄어듭니다.
- **NeverAsk**는 100회당 13.9건의 오류를 통과시킵니다. 검토를 아예 없애는 건 답이 아닙니다.

## 왜 필요한가

대부분의 에이전트 그래프는 검토 지점마다 `interrupt()`를 넣습니다. 첫날에는 안전하지만 곧 이렇게 됩니다.

- **검토 피로.** 문제없는 항목 수백 건을 본 사람은 보지 않고 승인하기 시작합니다(도장 찍기, rubber-stamping). 검토 단계는 비용은 그대로인데 오류를 더 이상 잡지 못합니다.
- **검토 단계를 없애는 건 위험합니다.** 어떤 항목은 정말로 사람이 봐야 하고, 그게 어떤 항목인지는 직감으로 알 수 없습니다.
- **항목별로, 근거를 가지고 판단하세요.** 대부분의 항목은 사람이 수백 번 수정 없이 승인한 상황에 속합니다. 나머지는 그렇지 않습니다. `hitl-gate`는 모든 검토 결정과 그 결과를 기록하고, 그 로그로 항목마다 물어볼 가치가 있는지 판단합니다.

## 30초 퀵스타트

```bash
pip install git+<this-repo-url>   # 아직 PyPI에 없음. 또는 clone 후 `pip install -e .`
```

```python
from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from hitl_gate import AlwaysAsk, DecisionLog, RulePolicy, add_gated_review, merge_dicts

class State(TypedDict, total=False):
    run_id: str
    hitl_decisions: Annotated[dict, merge_dicts]  # 게이트 결정이 저장되는 곳
    draft: str
    confidence: str

builder = StateGraph(State)
builder.add_node("draft", lambda s: {"draft": "Dear customer, ...", "confidence": "high"})
builder.add_node("send", lambda s: {})
builder.add_edge(START, "draft")

log = DecisionLog("decisions.jsonl")
gate = add_gated_review(
    builder, "review_draft", source="draft", target="send",
    policy=AlwaysAsk(),                                   # 안전하게 시작
    featurize=lambda s: {"confidence": s["confidence"]},  # 이산적이고 관측 가능한 신호
    make_review_payload=lambda s: {"draft": s["draft"]},  # 사람이 보게 될 내용
    apply_answer=lambda s, a: a.get("changes", {}),       # 사람의 수정 -> state 업데이트
    log=log,
)
builder.add_edge("send", END)
graph = builder.compile(checkpointer=MemorySaver())       # interrupt에는 checkpointer가 필요

# 나중에 로그에 근거가 쌓이면:
gate.policy = RulePolicy.fit(log.records, keys=["confidence"])
```

interrupt 재개는 평소 LangGraph 방식대로 하되, 게이트가 기대하는 답 형식을 씁니다.

```python
from langgraph.types import Command

graph.invoke(Command(resume={"outcome": "approve", "review_seconds": 12.0}), config)
# outcome: "approve" | "edit" | "retry" | "escalate";  선택적으로 "changes": {...}
```

`hitl_gate.sim.run_episode(graph, inputs, reviewer, thread_id)`가 invoke / interrupt / resume 루프를 스크립트 검토자로 대신 돌려 줍니다. 테스트와 시뮬레이션에 편리합니다.

## 동작 방식

```mermaid
flowchart LR
    S[source 노드] --> G{"gate 노드<br/>policy.decide()"}
    G -- ask --> R["review 노드<br/>interrupt()"] --> H((사람)) --> T[target 노드]
    G -- auto --> T
    G -. 기록 .-> L[(DecisionLog)]
    R -. 기록 + 결과 .-> L
    T -. 다운스트림 오류 .-> L
    L -- RulePolicy.fit / Q-learning --> P[더 나은 정책] --> G
```

`add_gated_review`는 `source`와 `target` 사이에 노드 두 개를 넣습니다.

- **`{gate_id}__gate`**: state에서 피처를 뽑고, 정책에 묻고, shadow 모드와 audit 샘플링을 적용한 뒤 결정을 `state["hitl_decisions"][gate_id]`에 저장합니다. 조건부 엣지로 라우팅합니다.
- **`{gate_id}__review`**: `interrupt({"gate_id", "reason", "payload"})`를 호출하고, 사람의 답을 적용한 뒤 결과를 기록합니다.

결정을 별도 노드에 두는 데는 이유가 있습니다. **LangGraph는 `interrupt()` 이후 재개할 때 interrupt를 호출한 노드를 처음부터 다시 실행합니다.** ask/auto 동전 던지기(audit 샘플링, 탐색)가 그 노드 안에 있으면 재개할 때 다시 굴려집니다. 여기서는 결정이 interrupt 전에 체크포인트되고, review 노드는 그 결정을 읽기만 합니다. 테스트 스위트가 이를 검증합니다.

그래프를 직접 연결한다면 `GateDecider(policy, mode, audit_rate, seed)`를 바로 쓰세요. state에 저장할 수 있는 평범한 dict를 반환하고, `make_record(run_id, gate_id, decision, answer)`가 로그 항목을 만듭니다.

## 정책

| 정책 | 물어보는 경우 | 학습 원천 |
|---|---|---|
| `AlwaysAsk` | 항상 (현재의 기본값) | — |
| `NeverAsk` | 절대 안 물음 (베이스라인 전용) | — |
| `RulePolicy.fit(records, keys, min_count=30, max_change_rate=0.02)` | 버킷 `(gate_id, 피처 값)`의 검토 샘플이 `min_count` 미만이거나 **또는** miss 비율이 `max_change_rate`를 넘을 때. 모르는 버킷 → ask. `describe()`가 규칙을 출력. | 기록된 검토 |
| `QTablePolicy` | `(gate_id, 이산화된 피처)`에서 `Q(ask) >= Q(auto)`일 때. 동점과 처음 보는 상태 → ask. 학습용 epsilon-greedy `act()`, `learn_episode()`, JSON `save()`/`load()`. | 직접 정의한 보상 (예: `-review_cost`, `-error_cost`) |

`name: str`과 `decide(gate_id, features) -> (ask: bool, reason: str)`를 가진 객체는 무엇이든 정책이 될 수 있습니다.

### 도장 찍기(rubber-stamp) 방지

"approve"라는 검토 결과가 항목이 괜찮았다는 증거는 아닙니다. `RulePolicy`는 사람이 결과를 바꿨거나, **또는** 승인했지만 나중에 다운스트림에서 실패한 경우(`DecisionLog.mark_downstream(run_id, gate_id, error=True)`)를 **miss**로 셉니다. 검토자가 전부 통과시키는데 나중에 계속 오류가 나는 버킷은 안전하다고 학습되지 않고 "ask"로 남습니다.

## 안전한 롤아웃

1. **Shadow.** `add_gated_review(..., mode="shadow")`: 사람에게는 항상 묻고, 정책이 *어떻게 했을지*를 로그에 남깁니다(`mode="shadow"`). 믿기 전에 정책이 auto로 보냈을 항목을 실제 결과와 비교하세요.
2. **audit를 곁들인 제한적 auto.** `audit_rate=0.05`로 라이브 전환: auto 결정의 5%는 여전히 사람에게 보내지고 `mode="audit"`로 기록됩니다. auto가 통과시키는 항목의 편향 없는 샘플을 얻을 수 있습니다.
3. **다운스트림 오류 시 롤백.** 다운스트림 실패를 로그에 표시하고 다시 fit하세요. 다운스트림에서 실패한 auto 통과 항목은 miss로 세므로, 해당 버킷은 다음 fit에서 "ask"로 돌아갑니다. `gate.policy = RulePolicy.fit(...)`로 정책을 그 자리에서 교체합니다.

## 로그 형식

게이트 결정 하나당 `DecisionRecord` 하나 (`DecisionLog(path)`를 주면 JSONL로도 저장):

| 필드 | 의미 |
|---|---|
| `run_id`, `gate_id` | 어떤 실행, 어떤 게이트 (`run_id`는 `state[run_id_key]`에서 읽음) |
| `features` | `featurize(state)`의 출력 |
| `decision` | **정책**의 결정: `"ask"` / `"auto"` |
| `policy_name`, `reason` | 어떤 정책이, 왜 |
| `mode` | `"live"`, `"shadow"`(항상 물음), `"audit"`(auto였지만 검토용으로 샘플링됨) |
| `human_outcome` | `"approve"`, `"edit"`, `"retry"`, `"escalate"`, 아무도 묻지 않았으면 `None` |
| `changed` | `human_outcome != "approve"` |
| `review_seconds` | 답에서 가져옴 |
| `downstream_error` | 나중에 `mark_downstream`으로 채움 |
| `ts` | 유닉스 시간 |

auto 통과는 gate 노드가 기록합니다. 물어본 항목은 사람이 답한 뒤에 기록되므로, 답이 오지 않은 검토는 기록이 남지 않습니다.

## 한계

- **시뮬레이션 결과는 가정에 좌우됩니다.** 그래프는 임의로 정한 오류율, 적발률(90%), 비용(검토 1회 = 1, 놓친 오류 1건 = 25)을 쓰는 장난감 파이프라인에서 나왔습니다. 여러분의 수치는 다를 것입니다. 먼저 실제 트래픽에서 shadow 모드로 돌려 보세요.
- **법적·계약상 필수인 승인은 절대 게이트하지 마세요.** 사람이 반드시 서명해야 한다면 `interrupt()`를 무조건 유지하세요.
- **피처는 이산적이어야 합니다.** 버킷과 Q-table은 피처당 몇 개의 값만 다룰 수 있습니다. 연속 신호는 `featurize`에서 구간화하세요(또는 `QTablePolicy(bins=...)`). 피처가 너무 많으면 버킷당 샘플이 부족해 모든 것이 "ask"로 남는데, 이는 안전한 방향의 실패입니다.
- **테이블 RL은 의도적으로 단순합니다.** `QTablePolicy`는 투명한 베이스라인이지 최신 학습기가 아닙니다. 예제에서는 시뮬레이션을 순수 파이썬으로 옮긴 빠른 미러에서 학습하며, 규칙 정책을 이기지 못합니다.
- `retry`와 `escalate`는 기록되지만 라우팅은 `review -> target`으로 고정입니다. 재시도 루프가 필요하면 `GateDecider`로 엣지를 직접 연결하세요.

## 관련 연구

- **Learning to Defer** (Madras, Pitassi & Zemel, NeurIPS 2018): 예측을 언제 사람 전문가에게 넘길지 결정하는 모델을 학습.
- **Ask4Help** (Singh et al., [arXiv 2211.09960](https://arxiv.org/abs/2211.09960)): 체화(embodied) 에이전트가 언제 전문가의 도움을 요청할지 정책을 학습.

## 개발

```bash
python3 -m venv .venv            # Python 3.14, langgraph 1.2에서 테스트됨
.venv/bin/pip install -e ".[dev,examples]"
.venv/bin/pytest -q
.venv/bin/python examples/document_pipeline/run.py   # 약 25초, docs/tradeoff.png 생성
```

예제는 시드가 고정되어 있어 재현 가능합니다.

## 라이선스

MIT. [LICENSE](LICENSE) 참조.
