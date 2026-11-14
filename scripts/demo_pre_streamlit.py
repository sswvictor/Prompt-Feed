"""Thesis demo: multi-turn LLM assistant with M1/M2 intention tracking + M3 privacy memory."""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.m1_nlu.factory import get_extractor
from src.m1_nlu.types import StructuredTaskVariable, normalize_stv
from src.m2_reasoning.belief import entropy, initialize_belief, uniform_prior, update_belief
from src.m2_reasoning.convergence import (
    incorporate_feedback,
    is_converged,
    k_schedule,
    top1_probability,
    top_k_hypothesis_ids,
)
from src.m2_reasoning.hypothesis import GoalHypothesis, generate_hypothesis_space
from src.m3_memory.profile import AggregatedUserProfile
from src.m3_memory.store import InMemoryStore
from src.dialogue import ChatMessage, DialogueOrchestrator
from src.config.env import gpt_connection_status, load_project_env
from src.pipeline.config import PipelineConfig, pipeline_config_from_yaml
from src.pipeline.runner import PipelineResult, PipelineRunner
from src.pipeline.session import SessionManager, SessionState

FF_SCENARIOS: Dict[str, List[str]] = {
    "Vague wellness": [
        "I don't know, maybe something for body and mood, tonight maybe",
        "Actually weekend, and not too far",
        "Something relaxing is fine",
    ],
    "Conflict time": [
        "Need a place now but also this weekend, close to office",
        "Let's do weekend instead",
        "Fitness or wellness, quick session",
    ],
    "Clear fitness": [
        "Book a quick workout class nearby tonight",
    ],
}

# --- M1: utterance presets ---
M1_PRESETS: Dict[str, str] = {
    "Ambiguous": "Medical or wellness, not sure, but I need help now",
    "Conflicting time": "Need a place now but also this weekend, close to office",
    "Clear intent": "Book a quick workout class nearby tonight",
}

UNCERTAIN_CUES = ("not sure", "maybe", "don't know", " or ", "i don't know")
CONFLICT_CUES = (("now", "weekend"), ("weekend", "now"))

# --- M2: evidence (STV) presets — fed directly, independent of live M1 ---
def _stv(
    category: str,
    conf: float,
    temporal: str | None = None,
    urgency: str = "medium",
    location: str | None = None,
    tags: List[str] | None = None,
    raw: str = "preset",
) -> StructuredTaskVariable:
    stv = StructuredTaskVariable(
        service_category=category,
        category_confidence=conf,
        temporal_preference=temporal,
        urgency_level=urgency,
        time_window_days=1 if urgency == "high" else 7,
        location_context=location,
        sentiment_tags=tags or [],
        extraction_confidence=conf,
        raw_input=raw,
        clarification_needed=False,
    )
    return normalize_stv(stv)


M2_STV_PRESETS: Dict[str, StructuredTaskVariable] = {
    "Ambiguous evidence": _stv(
        "ambiguous",
        0.35,
        temporal=None,
        urgency="medium",
        raw="ambiguous preset",
    ),
    "Clear fitness (evening, nearby)": _stv(
        "fitness",
        0.92,
        temporal="evening",
        urgency="medium",
        location="nearby",
        tags=["quick"],
        raw="clear fitness preset",
    ),
    "Wellness + relaxing": _stv(
        "wellness",
        0.88,
        temporal="weekend",
        urgency="low",
        location="nearby",
        tags=["relaxing"],
        raw="wellness preset",
    ),
}

# Full space (~360) so all service categories appear — max_size=80 only contained fitness.
HYPOTHESES: List[GoalHypothesis] = generate_hypothesis_space(max_size=360)
PRIOR = uniform_prior(HYPOTHESES)
INIT_BELIEF = initialize_belief(HYPOTHESES)
BY_ID = {h.hypothesis_id: h for h in HYPOTHESES}
CONVERGE_THRESHOLD = 0.85


def _apply_demo_uncertainty(stv: StructuredTaskVariable, text: str) -> StructuredTaskVariable:
    lower = text.lower()
    if any(cue in lower for cue in UNCERTAIN_CUES):
        stv.category_confidence = min(stv.category_confidence, 0.35)
    for a, b in CONFLICT_CUES:
        if a in lower and b in lower:
            stv.category_confidence = min(stv.category_confidence, 0.38)
    return normalize_stv(stv)


def format_hypothesis(h: GoalHypothesis) -> str:
    return (
        f"{h.service_category} · {h.service_subtype} · "
        f"{h.temporal_slot} · {h.location_zone} · {h.price_tier}"
    )


def category_posterior(belief: Dict[str, float]) -> Dict[str, float]:
    cats: Dict[str, float] = {}
    for h_id, p in belief.items():
        cat = BY_ID[h_id].service_category
        cats[cat] = cats.get(cat, 0.0) + p
    total = sum(cats.values()) or 1.0
    return {k: v / total for k, v in sorted(cats.items(), key=lambda x: x[1], reverse=True)}


def run_m2(stv: StructuredTaskVariable, turn: int) -> Tuple[Dict[str, float], List[str]]:
    belief = update_belief(INIT_BELIEF, stv, PRIOR, HYPOTHESES)
    top_ids = top_k_hypothesis_ids(belief, turn)
    return belief, top_ids


def top_k_rows(belief: Dict[str, float], top_ids: List[str]) -> List[Dict[str, object]]:
    rows = []
    for rank, h_id in enumerate(top_ids, start=1):
        h = BY_ID[h_id]
        rows.append(
            {
                "rank": rank,
                "belief_pct": round(belief[h_id] * 100, 2),
                "goal": format_hypothesis(h),
                "id": h_id,
            }
        )
    return rows


def best_goal_per_category(belief: Dict[str, float]) -> List[Dict[str, object]]:
    best: Dict[str, Tuple[str, float]] = {}
    for h_id, p in belief.items():
        cat = BY_ID[h_id].service_category
        if cat not in best or p > best[cat][1]:
            best[cat] = (h_id, p)
    rows = []
    for cat in sorted(best.keys()):
        h_id, p = best[cat]
        rows.append(
            {
                "category": cat,
                "belief_pct": round(p * 100, 2),
                "best_goal": format_hypothesis(BY_ID[h_id]),
            }
        )
    return rows


def render_m1_tab() -> None:
    st.header("Step 1 — Language extraction (M1)")
    st.caption("Maps utterances to structured evidence (STV). Does not rank bookable goals.")

    if "m1_text" not in st.session_state:
        st.session_state.m1_text = M1_PRESETS["Ambiguous"]

    text = st.text_area("User utterance", value=st.session_state.m1_text, height=72)

    cols = st.columns(len(M1_PRESETS))
    for col, (label, preset) in zip(cols, M1_PRESETS.items()):
        with col:
            if st.button(label, key=f"m1_{label}", use_container_width=True):
                st.session_state.m1_text = preset
                st.rerun()

    if st.button("Extract STV", type="primary", key="m1_extract"):
        st.session_state.m1_text = text.strip() or M1_PRESETS["Ambiguous"]

    utterance = st.session_state.m1_text
    extractor = get_extractor("pipeline")
    stv = _apply_demo_uncertainty(extractor.extract(utterance), utterance)

    if stv.clarification_needed or stv.service_category == "ambiguous":
        st.warning("Clarification needed — M1 will not commit to a single category.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category", stv.service_category)
    c2.metric("Confidence", f"{stv.category_confidence:.2f}")
    c3.metric("When", stv.temporal_preference or "—")
    c4.metric("Urgency", stv.urgency_level)

    c5, c6 = st.columns(2)
    c5.metric("Location", stv.location_context or "—")
    tags = ", ".join(stv.sentiment_tags) if stv.sentiment_tags else "—"
    c6.metric("Tags", tags)

    with st.expander("Raw STV (debug)"):
        st.json(asdict(stv))

    st.info(
        "M1 stops here. In the **M2 tab**, the same story is modeled as belief over "
        "**full bookable goals** (category + subtype + time + place + price), not another category guess."
    )


def render_m2_tab() -> None:
    st.header("Step 2 — Belief over bookable goals (M2)")
    st.caption(
        "Maintains P(goal | evidence) over ~80 hypotheses. Top-K narrows candidates; "
        "M2 ranks **bookable goals** (category + subtype + time + place + price). "
        "LLM handles natural replies using the intention prompt block."
    )

    if "m2_preset" not in st.session_state:
        st.session_state.m2_preset = "Ambiguous evidence"
    if "m2_turn" not in st.session_state:
        st.session_state.m2_turn = 0
    if "m2_belief" not in st.session_state:
        st.session_state.m2_belief = None
    if "m2_top_ids" not in st.session_state:
        st.session_state.m2_top_ids = []
    if "m2_feedback" not in st.session_state:
        st.session_state.m2_feedback = False
    if "m2_before" not in st.session_state:
        st.session_state.m2_before = None
    if "m2_cache_key" not in st.session_state:
        st.session_state.m2_cache_key = None

    st.caption("Prior: **uniform** (enable M3 tab for history-adjusted prior)")

    preset_name = st.selectbox(
        "Evidence (STV preset)",
        list(M2_STV_PRESETS.keys()),
        index=list(M2_STV_PRESETS.keys()).index(st.session_state.m2_preset),
        key="m2_preset_select",
    )
    st.session_state.m2_preset = preset_name

    turn = st.slider("Dialogue turn", 0, 3, st.session_state.m2_turn, key="m2_turn_slider")
    st.session_state.m2_turn = turn

    stv = M2_STV_PRESETS[st.session_state.m2_preset]
    cache_key = (st.session_state.m2_preset, turn)

    if st.session_state.m2_cache_key != cache_key:
        st.session_state.m2_cache_key = cache_key
        st.session_state.m2_feedback = False
        st.session_state.m2_before = None

    if st.button("Update belief", type="primary", key="m2_update") or st.session_state.m2_belief is None:
        belief, top_ids = run_m2(stv, turn)
        st.session_state.m2_belief = belief
        st.session_state.m2_top_ids = top_ids

    belief = st.session_state.m2_belief
    top_ids = st.session_state.m2_top_ids
    k = k_schedule(turn)
    reduction = (len(HYPOTHESES) - k) / len(HYPOTHESES) if HYPOTHESES else 0.0

    with st.expander("STV evidence used by M2 (read-only)"):
        st.json(asdict(stv))

    st.subheader("Candidate funnel")
    f1, f2, f3 = st.columns(3)
    f1.metric("Hypothesis space |H|", len(HYPOTHESES))
    f2.metric(f"Top-K shown (turn {turn})", k)
    f3.metric("Reduction vs |H|", f"{reduction * 100:.0f}%")

    ent = entropy(belief)
    top1 = top1_probability(belief)
    converged = is_converged(belief, threshold=CONVERGE_THRESHOLD)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Entropy", f"{ent:.2f}")
    m2.metric("Top-1 hypothesis", f"{top1 * 100:.1f}%")
    m3.metric("Converged (>0.85)", "Yes" if converged else "No")
    m4.metric("Clarify gate (M1)", "Yes" if stv.clarification_needed else "No")

    st.subheader("Top bookable goals")
    st.dataframe(
        top_k_rows(belief, top_ids),
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "belief_pct": st.column_config.NumberColumn("Belief %", format="%.2f"),
            "goal": st.column_config.TextColumn("Goal (category · subtype · time · place · price)"),
            "id": st.column_config.TextColumn("ID", width="small"),
        },
    )

    st.caption(
        "Low Top-1 % is expected: probability spreads across many granular goals "
        "(subtype × time × place × price), not only the category label."
    )

    st.subheader("Strongest goal per category")
    st.dataframe(
        best_goal_per_category(belief),
        use_container_width=True,
        hide_index=True,
    )
    if stv.service_category == "ambiguous":
        st.info(
            "Ambiguous evidence: category mass stays balanced; M2 keeps multiple bookable paths "
            "open until the user clarifies or picks a goal."
        )

    if st.button("User picks rank #1", use_container_width=True, key="m2_pick"):
        if top_ids and not st.session_state.m2_feedback:
            st.session_state.m2_before = {"top1": top1, "entropy": ent}
            st.session_state.m2_belief = incorporate_feedback(belief, top_ids[0])
            st.session_state.m2_top_ids = top_k_hypothesis_ids(
                st.session_state.m2_belief, turn
            )
            st.session_state.m2_feedback = True
            st.rerun()

    if st.session_state.m2_feedback and st.session_state.m2_before:
        before = st.session_state.m2_before
        after_belief = st.session_state.m2_belief
        after_top1 = top1_probability(after_belief)
        after_ent = entropy(after_belief)
        b1, b2, b3 = st.columns(3)
        b1.metric("Top-1 before", f"{before['top1'] * 100:.1f}%")
        b2.metric(
            "Top-1 after",
            f"{after_top1 * 100:.1f}%",
            delta=f"{(after_top1 - before['top1']) * 100:+.1f} pts",
        )
        b3.metric(
            "Entropy after",
            f"{after_ent:.2f}",
            delta=f"{after_ent - before['entropy']:.2f}",
            delta_color="inverse",
        )
        st.success("User feedback sharpened belief — multi-turn narrowing.")

    with st.expander("Category summary (debug only — not the M2 product output)"):
        for cat, prob in category_posterior(belief).items():
            st.progress(min(1.0, prob * 2.5), text=f"{cat} — {prob * 100:.1f}%")


@st.cache_resource
def _m3_demo_store() -> InMemoryStore:
    import json

    store = InMemoryStore()
    path = ROOT / "data" / "profiles" / "synthetic_users.json"
    if path.exists():
        for row in json.loads(path.read_text(encoding="utf-8")):
            store.save_profile(AggregatedUserProfile.from_dict(row))
    return store


def render_m3_tab() -> None:
    st.header("Step 3 — Privacy memory prior (M3)")
    st.caption("Aggregate history adjusts M2 prior only — never injected into M1 or LLM prompts.")

    store = _m3_demo_store()
    personas = store.list_user_ids() or ["persona_cold_start"]
    persona = st.selectbox("User persona (aggregated profile)", personas, key="m3_persona")

    utterance = st.text_input(
        "Utterance for full pipeline (M1→M3→M2)",
        value="I need a relaxing spa nearby this weekend",
        key="m3_utterance",
    )

    cfg = PipelineConfig(
        enable_m3=True,
        user_id=persona,
        memory_backend="inmemory",
        hypothesis_max_size=360,
    )
    runner = PipelineRunner(cfg, store=store)
    cmp = runner.prior_comparison(persona)
    summary = store.get_summary(persona)

    c1, c2, c3 = st.columns(3)
    c1.metric("Uniform prior entropy", f"{cmp['uniform_entropy']:.2f}")
    c2.metric("History prior entropy", f"{cmp['history_entropy']:.2f}")
    c3.metric("Total sessions", summary.get("total_sessions", 0))

    with st.expander("Aggregated profile (transparency view)"):
        st.json(summary)

    if st.button("Run full pipeline", type="primary", key="m3_run"):
        session = SessionState(session_id="demo", user_id=persona)
        mgr = SessionManager(cfg, runner=runner)
        result = mgr.process_turn(session, utterance)
        st.subheader("Pipeline result")
        m1, m2, m3 = st.columns(3)
        m1.metric("M1 category", result.stv.service_category)
        m2.metric("Posterior entropy", f"{result.posterior_entropy:.2f}")
        m3.metric("Top-1 prob", f"{result.top1_prob * 100:.1f}%")


def _ff_init_state() -> None:
    if "ff_chat_messages" not in st.session_state:
        st.session_state.ff_chat_messages = []
    if "ff_session" not in st.session_state:
        st.session_state.ff_session = None
    if "ff_persona_prev" not in st.session_state:
        st.session_state.ff_persona_prev = None
    if "ff_m3_prev" not in st.session_state:
        st.session_state.ff_m3_prev = None
    if "ff_inject_prev" not in st.session_state:
        st.session_state.ff_inject_prev = None
    if "ff_entropy_history" not in st.session_state:
        st.session_state.ff_entropy_history = []


def _ff_reset_chat(persona: str) -> None:
    st.session_state.ff_chat_messages = []
    st.session_state.ff_entropy_history = []
    mgr_cfg = _ff_build_config(
        persona,
        st.session_state.get("ff_m3_enabled", True),
        st.session_state.get("ff_inject_enabled", True),
    )
    mgr = SessionManager(mgr_cfg, runner=_ff_build_runner(mgr_cfg))
    st.session_state.ff_session = mgr.new_session(persona)


def _ff_build_config(persona: str, enable_m3: bool, inject_intention: bool) -> PipelineConfig:
    base = pipeline_config_from_yaml()
    dialogue = replace(base.dialogue, inject_intention_block=inject_intention)
    return PipelineConfig(
        extractor_mode=base.extractor_mode,
        enable_m3=enable_m3,
        user_id=persona,
        memory_backend="inmemory",
        hypothesis_max_size=base.hypothesis_max_size,
        converge_threshold=base.converge_threshold,
        m3_weight_when_ambiguous=base.m3_weight_when_ambiguous,
        config_path=base.config_path,
        dialogue=dialogue,
    )


def _ff_build_runner(cfg: PipelineConfig) -> PipelineRunner:
    store = _m3_demo_store() if cfg.enable_m3 else None
    return PipelineRunner(cfg, store=store)


def _render_pipeline_details(result: PipelineResult, intention_block=None, msg: ChatMessage | None = None) -> None:
    m0, m1, m2, m3 = st.columns(4)
    if msg is not None:
        m0.metric("Mode", msg.session_mode)
        m0.caption(f"Surface: {msg.surface_policy}")
    m1.metric("M1 category", result.stv.service_category)
    m2.metric("Category top-1", f"{result.top1_category_prob * 100:.1f}%")
    m3.metric("Hypothesis top-1", f"{result.top1_prob * 100:.1f}%")
    st.metric("Posterior entropy", f"{result.posterior_entropy:.2f}")

    with st.expander("STV evidence"):
        st.json(asdict(result.stv))

    if intention_block is not None:
        with st.expander("Intention block → LLM", expanded=False):
            st.code(intention_block.to_prompt_text())
            st.json(intention_block.privacy_audit())

    st.markdown("**Top-K bookable goals**")
    by_id = {h.hypothesis_id: h for h in result.hypotheses}
    hyp_rows = []
    for rank, h_id in enumerate(result.top_k_ids[:8], start=1):
        h = by_id[h_id]
        hyp_rows.append(
            {
                "rank": rank,
                "belief_pct": round(result.posterior[h_id] * 100, 2),
                "goal": format_hypothesis(h),
            }
        )
    st.dataframe(hyp_rows, use_container_width=True, hide_index=True)


def render_assistant_tab() -> None:
    st.header("Multi-turn Assistant")
    st.caption(
        "LLM replies with an intention prompt block (M1→M3→M2). "
        "M3 aggregate profile stays local and is **not** sent to the LLM."
    )

    _ff_init_state()
    store = _m3_demo_store()
    personas = store.list_user_ids() or ["persona_cold_start"]

    top1, top2, top3, top4 = st.columns([2, 1, 1, 1])
    with top1:
        persona = st.selectbox("User persona", personas, key="ff_persona")
    with top2:
        enable_m3 = st.checkbox("M3 memory", value=True, key="ff_m3")
    with top3:
        inject_block = st.checkbox("Intention block", value=True, key="ff_inject")
    with top4:
        if st.button("New chat", key="ff_new_chat"):
            st.session_state.ff_m3_enabled = enable_m3
            st.session_state.ff_inject_enabled = inject_block
            _ff_reset_chat(persona)
            st.rerun()

    st.session_state.ff_m3_enabled = enable_m3
    st.session_state.ff_inject_enabled = inject_block
    persona_changed = st.session_state.ff_persona_prev != persona
    m3_changed = st.session_state.ff_m3_prev != enable_m3
    inject_changed = st.session_state.ff_inject_prev != inject_block
    if persona_changed or m3_changed or inject_changed or st.session_state.ff_session is None:
        _ff_reset_chat(persona)
        st.session_state.ff_persona_prev = persona
        st.session_state.ff_m3_prev = enable_m3
        st.session_state.ff_inject_prev = inject_block

    cfg = _ff_build_config(persona, enable_m3, inject_block)
    runner = _ff_build_runner(cfg)
    session: SessionState = st.session_state.ff_session
    orchestrator = DialogueOrchestrator(
        cfg,
        session_manager=SessionManager(cfg, runner=runner),
        inject_intention_block=inject_block,
    )

    scen_cols = st.columns(len(FF_SCENARIOS))
    for col, (label, turns) in zip(scen_cols, FF_SCENARIOS.items()):
        with col:
            if st.button(f"Scenario: {label}", key=f"sc_{label}", use_container_width=True):
                _ff_reset_chat(persona)
                session = st.session_state.ff_session
                orchestrator = DialogueOrchestrator(
                    cfg,
                    session_manager=SessionManager(cfg, runner=runner),
                    inject_intention_block=inject_block,
                )
                for t in turns:
                    tr = orchestrator.handle_turn(session, t, st.session_state.ff_chat_messages)
                    for cm in tr.to_chat_messages(t):
                        st.session_state.ff_chat_messages.append(cm)
                    if tr.pipeline_result is not None:
                        st.session_state.ff_entropy_history.append(
                            tr.pipeline_result.posterior_entropy
                        )
                st.rerun()

    chat_col, side_col = st.columns([3, 2])

    with side_col:
        gpt_status = gpt_connection_status(
            model=cfg.dialogue.model,
            enable_gpt=cfg.dialogue.enable_gpt,
        )
        if gpt_status["connected"]:
            st.success(f"Dialogue: GPT ({gpt_status['model']})")
        else:
            st.warning(f"Dialogue: Mock — {gpt_status['reason']}")

        st.subheader("Intention & privacy")
        if enable_m3:
            cmp = runner.prior_comparison(persona)
            summary = store.get_summary(persona)
            st.metric("History prior entropy", f"{cmp['history_entropy']:.2f}")
            st.metric("Dialogue turn", session.turn)
            with st.expander("M3 profile (NOT sent to LLM)", expanded=False):
                st.json(summary)
        if st.session_state.ff_entropy_history:
            st.line_chart(
                {"posterior_entropy": st.session_state.ff_entropy_history},
                height=120,
            )
        if st.button("End session → update M3 aggregates", key="ff_end_session"):
            if orchestrator.end_session_with_outcome(session):
                st.success("Aggregate counts updated (no raw text stored).")
            else:
                st.warning("Enable M3 and complete at least one task turn first.")

    with chat_col:
        max_turns = cfg.dialogue.max_history_turns
        for msg in st.session_state.ff_chat_messages[-max_turns * 2 :]:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
                if msg.role == "assistant" and msg.pipeline_result is not None:
                    with st.expander("Tracking details"):
                        _render_pipeline_details(msg.pipeline_result, msg.intention_block, msg)

        user_input = st.chat_input("Ask about a booking or say hello…")
        if user_input:
            turn_result = orchestrator.handle_turn(
                session,
                user_input,
                chat_history=st.session_state.ff_chat_messages,
            )
            if turn_result.pipeline_result is not None:
                st.session_state.ff_entropy_history.append(
                    turn_result.pipeline_result.posterior_entropy
                )
            for chat_msg in turn_result.to_chat_messages(user_input):
                st.session_state.ff_chat_messages.append(chat_msg)
            st.rerun()


def main() -> None:
    load_project_env()
    st.set_page_config(page_title="Booking Assistant Demo", layout="wide")
    st.title("Ethical AI Booking Assistant")
    st.caption("Multi-turn intention tracking + privacy memory · LLM surface")

    tab_assistant, tab_m1, tab_m2, tab_m3 = st.tabs(
        [
            "Multi-turn Assistant",
            "Debug — M1",
            "Debug — M2",
            "Debug — M3",
        ]
    )

    with tab_assistant:
        render_assistant_tab()

    with tab_m1:
        render_m1_tab()

    with tab_m2:
        render_m2_tab()

    with tab_m3:
        render_m3_tab()


if __name__ == "__main__":
    main()
