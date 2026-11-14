# Multi-turn Clarify + Topic Transition — Design Spec (Draft for Review)

**Date:** 2026-07-11  
**Status:** Implemented (2026-07-11)  
**Defaults locked from discussion:** `transition_observe_turns = 3`; hard switches must be captured; soft switches must not over-fire on casual chat.

---

## 1. Goals

| Goal | Mechanism |
|------|-----------|
| Resolve vague intent within one topic | **Clarify track** (ask / enumerate / confirm) |
| Handle explicit topic shift without breaking rhythm | **Transition observe** (dual belief, soft keyword replies, promote after 3 turns) |
| Keep M2 belief accurate | Dual-track decay + accumulation; promote or fall back to clarify |
| Avoid false switches on small talk | **Hard triggers only** for transition; soft shift disabled or very strict |

---

## 2. Session modes (single hierarchy)

```
TRACKING          # single active belief track
  ├─ CLARIFY      # same-topic uncertainty (surface: ask / enumerate)
  └─ SUGGEST      # converged or user-confirmed

TRANSITION        # dual-track observation (surface: observe, NOT clarify)
  └─ observe_turn ∈ [1..3]
```

**Invariant:** `CLARIFY` and `TRANSITION` are mutually exclusive at the policy layer.

---

## 3. Clarify design (recap + final rules)

### 3.1 When to enter CLARIFY

Enter when **all** hold:

- `session.mode ∈ {TRACKING}` (not in TRANSITION)
- M2 policy from `resolve_dialogue_policy()` → `clarify`
  - `force_clarify` (emotion escape), or
  - `category_entropy ≥ 0.85 × log(6)`, or
  - not `converged` and `top1_category_prob < 0.5`

### 3.2 Clarify sub-state

```python
clarify_active: bool
clarify_offered_categories: list[str]   # top 2-3 from category_masses
pending_confirmation: str | None      # category awaiting yes/no
clarify_last_prompt_type: str           # enumerate | confirm | slot
```

### 3.3 Clarify surface behaviors

| Step | Condition | Surface (GPT) | Exit |
|------|-----------|---------------|------|
| Enumerate | first clarify turn, no pending | List top 2–3 **categories** from M2 | set `offered_categories` |
| Confirm | user picked / strong keyword | Short confirm: "So you want {cat}?" | set `pending_confirmation` |
| User confirm | yes / 对 / 就这个 / sure (with history) | Suggest | clear clarify → SUGGEST |
| User gives keyword | explicit category in utterance | Suggest + boost that category | clear clarify → SUGGEST |
| Still vague after 2 clarify cycles | — | Re-enumerate or slot ask | stay CLARIFY |

### 3.4 Router fix (required for clarify)

- Pass `chat_history` into router (fix current bug).
- If `clarify_active` or `pending_confirmation`:
  - Short replies (`yes`, `no`, `the second one`, `dining`) → **force TASK**, never chitchat.

### 3.5 Clarify exit conditions (any one)

1. User **confirms** `pending_confirmation`
2. User **provides explicit category keyword** (rule > M1 if conflict, see §6)
3. M2 **numeric**: `converged` or `top1_category_prob ≥ 0.5` **and** not contradicted by user text

---

## 4. Transition design (recap + final rules)

### 4.1 Philosophy

> User explicitly shifts topic → **do not clarify by default**; use **keyword-grounded soft replies** for 3 turns while dual tracks update; then promote, abort, or fall back to clarify.

### 4.2 Triggers — hard only (casual-insensitive)

**Open TRANSITION** when utterance matches **hard trigger** (rule-based, high precision):

| Pattern class | Examples (EN) | Examples (ZH) |
|---------------|---------------|---------------|
| Abandon | forget it, never mind, skip that, not that | 算了, 不要了, 不用了, 换个别的话题 |
| Explicit pivot | instead, what about, rather, switch to | 还是…吧, 换成, 不如 |
| Abandon + cue | forget it … nail / instead … travel | 算了我想吃饭 |

**Implementation:** regex/keyword list, case-insensitive; require abandon **or** pivot marker.

**Soft shift (M1 category jump) — OFF by default**

- Do **not** open TRANSITION on `M1.category ≠ active_category` alone (too sensitive for chitchat / M1 noise).
- Optional later: enable only if `conf ≥ 0.7` **and** 2 consecutive turns (feature flag).

### 4.3 Hard trigger must not miss

Maintain `HARD_TRANSITION_PATTERNS` in one module; unit tests per pattern class.

On hard trigger:

```python
session.mode = TRANSITION
session.observe_turn = 0
session.transition_trigger = "hard"
active_track → decay_mode
candidate_track → init from utterance keywords (§6), not M1 alone
clarify_active = False
pending_confirmation = None
```

### 4.4 Observe window (3 turns)

Each user turn while `mode == TRANSITION`:

1. Update **active** belief (decay if no support for active_category)
2. Update **candidate** belief (normal M2 update)
3. Increment `observe_turn`
4. Surface: **OBSERVE** prompt (keyword-rich, non-interrogative) — **never CLARIFY**
5. Log `TransitionTrace` row

### 4.5 Observe surface (GPT)

System constraints:

- Include ≥1 **user keyword** + ≥1 **candidate-class synonym**
- 2–3 sentences; open ending
- **Forbidden:** "Which do you mean?", "Please confirm", forced A/B/C choice
- Allowed: light suggestions, "if you want X…", "you could also…"

### 4.6 Decision after turn 3 (or early promote)

**Promote** (candidate becomes sole track) if:

```
candidate_top1_cat ≥ 0.45
AND candidate_top1_cat ≥ active_top1_cat + 0.15
AND candidate mass not falling over last 2 turns
```

**Early promote** (before turn 3) if:

- User utterance has **strong category keyword** (rule list) with clear mapping, or
- `candidate_top1_cat ≥ 0.55` after turn 1

**Abort transition** (return to TRACKING on active) if:

- User provides **support for active_category** (keyword or M1 conf ≥ 0.6 on active)
- Active mass rises above candidate for 2 consecutive turns

**Fall back to CLARIFY** if:

- After turn 3, promotion and abort both fail (tracks still overlapping)
- Then allow enumerate / confirm on **merged top 2 categories**

### 4.7 Candidate replacement (same window)

If hard trigger again with **new** category cue while already in TRANSITION:

- Replace `candidate_track` (do not stack a third track)
- Reset `observe_turn = 0`
- Active continues decay

---

## 5. Conflict matrix: Clarify vs Transition

| Situation | Clarify wants | Transition wants | **Resolution** |
|-----------|---------------|------------------|----------------|
| User says "forget it, nail" | M2 uncertain → clarify | Hard trigger → observe | **Transition wins**; suppress clarify |
| User vague on one topic | clarify enumerate | — | **Clarify only** (TRACKING) |
| In TRANSITION, user says "yes" | confirm pending | weak boost candidate | **No pending in TRANSITION** → treat as weak candidate boost, observe reply |
| In CLARIFY, user says "forget it" | — | hard trigger | **Exit clarify** → open TRANSITION |
| In SUGGEST, user hard pivot | — | hard trigger | **Open TRANSITION**; decay old suggest track |
| Chitchat "how are you" | router → chitchat | — | **Chitchat**; no mode change (no soft shift) |
| M1 flips travel→medical one turn | could clarify | soft shift (disabled) | **Stay TRACKING**; single-track M2 only |
| After 3 observe turns, still tied | clarify fallback | end observe | **Enter CLARIFY** on top-2 categories |

**Priority order (highest first):**

1. `emotion_escape` → force clarify (even in TRANSITION — exceptional)
2. `hard_transition_trigger` → TRANSITION observe
3. `clarify_active` + short answer → clarify confirm path
4. M2 `resolve_dialogue_policy` → clarify vs suggest
5. Router chitchat (only if none of above)

---

## 6. Keyword authority (shared by both schemes)

When **rule keyword category** conflicts with **M1 category**:

```
priority: explicit_user_keyword > pending_confirmation > candidate_init > M1 > M2 prior
```

Keyword lists (minimal v1):

- travel: travel, flight, hotel, trip, airport
- dining: eat, hungry, restaurant, food, lunch, dinner, meal
- beauty: nail, manicure, skin, salon, beauty
- fitness: gym, workout, fitness, yoga, class
- wellness: spa, massage, relax, unwind
- medical: doctor, clinic, checkup, medical, health

Use for: candidate init, early promote, clarify exit, observe reply injection.

---

## 7. SessionState fields (implementation checklist)

```python
# Mode
mode: Literal["TRACKING", "TRANSITION"]
surface_policy: Literal["clarify", "suggest", "observe", "chitchat"]

# Active track
active_category: str | None
belief: dict[str, float]

# Clarify (TRACKING only)
clarify_active: bool
clarify_offered_categories: list[str]
pending_confirmation: str | None

# Transition (TRANSITION only)
candidate_belief: dict[str, float] | None
candidate_seed_category: str | None
observe_turn: int  # 0..3
transition_trigger: str
transition_traces: list[dict]

# Existing
force_clarify, stv_history, turn, user_id, ...
```

---

## 8. Component change list (for implementation)

| File / area | Change |
|-------------|--------|
| `src/pipeline/session.py` | Mode fields; dual-track update; decay; promote/abort |
| `src/dialogue/transition.py` | **NEW** hard trigger detector, keyword authority, TransitionTrace |
| `src/dialogue/clarify.py` | **NEW** clarify sub-state machine, offered categories |
| `src/dialogue/intention_context.py` | `surface_policy` respects mode; suppress clarify in TRANSITION |
| `src/dialogue/orchestrator.py` | Fix router history; orchestrate mode transitions; observe/clarify prompts |
| `src/dialogue/router.py` | History-aware; clarify/transition overrides |
| `src/dialogue/task_reply.py` | `_OBSERVE_SYSTEM`, `_CLARIFY_ENUM_SYSTEM` variants |
| `scripts/demo_pre_streamlit.py` | Show mode, active vs candidate mass, observe_turn |
| `tests/dialogue/` | Hard triggers, yes-after-clarify, transition 3-turn promote, no soft-shift on chitchat |

---

## 9. Evaluation hooks

| Metric | Definition |
|--------|------------|
| `clarify_resolution_turn` | First turn exiting CLARIFY to SUGGEST |
| `transition_promote_turn` | Observe turn when promote fires (1–3) |
| `transition_abort_rate` | User returns to old topic |
| `false_transition_rate` | Hard trigger on non-shift utterances (should ≈ 0) |
| `post_switch_gold_match` | Scenario gold category after promote |

Add scenario rows with `expected_transition_at_turn` in `eval_scenarios.json` (later).

---

## 10. Defaults (locked for v1)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `transition_observe_turns` | **3** | User preference; enough for trend |
| `soft_shift_enabled` | **false** | Casual-insensitive |
| `hard_triggers` | **on** | Must capture explicit pivots |
| `decay_factor` | 0.75 / turn | Active track without support |
| `θ_promote` | 0.45 | Candidate absolute |
| `δ_promote` | 0.15 | Candidate vs active margin |
| `clarify_enumerate_k` | 3 | Top categories to offer |
| `emotion_escape` | overrides observe → clarify | Safety |

---

## 11. Open questions for review meeting

1. **Emotion escape during TRANSITION:** spec says force clarify — agree?
2. **Same-class slot change (spa→massage):** no TRANSITION — agree?
3. **After 3 turns fall back to CLARIFY:** enumerate top-2 from **both** tracks or candidate only?
4. **M3 prior on candidate init:** use history prior vs uniform — recommend history prior.
5. **LLM observe vs template observe** when GPT offline — template fallback with keywords?

---

## 12. Execution order (post-review)

```
Phase 1 — P0 fixes
  - Router chat_history + clarify short-answer force TASK
  - SessionState mode skeleton

Phase 2 — Clarify FSM
  - enumerate / confirm / exit
  - tests

Phase 3 — Transition hard triggers + 3-turn observe
  - dual-track update, decay, promote/abort
  - observe prompt
  - tests

Phase 4 — Demo + scenario eval
  - UI metrics (category-level, mode badge)
  - 2–3 multi-topic scenarios
```

---

## Changelog

| Date | Note |
|------|------|
| 2026-07-11 | Initial unified spec from clarify + transition discussion |
