"""Defence recording UI: you type; the sidecar tracks.

Run from Code/:
    streamlit run scripts/demo_defence.py

Collapse the sidebar before recording. Keep the three lines on paper.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from scripts.defence_dining import DefenceDiningRunner, surface_reply
from src.config.env import gpt_connection_status, load_project_env
from src.dialogue.types import ChatMessage, DialogueTurnType

BEATS = (
    "Wait for clarify. Glance at Policy and H_cat.",
    "Wait for suggest. H_cat should fall.",
    "Open Intention block → LLM. Point at Withheld.",
)

ASSETS = ROOT / "artifacts" / "demo"
FIX_ICON = ASSETS / "fixmeapp.png"
SS_ICON = ASSETS / "shuwei.png"
CATEGORIES = ("beauty", "dining", "fitness", "medical", "travel", "wellness")

NAVY = "#0A1226"
INK = "#182E63"
ACCENT = "#356AE9"
KTH = "#004791"
SAND = "#EBE5E0"
CREAM = "#F6F1EA"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;1,600&family=Poppins:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: "Poppins", sans-serif; }}
        #MainMenu, header, footer, .stDeployButton {{ visibility: hidden; }}
        .stApp {{ background: {CREAM}; color: {NAVY}; }}
        .block-container {{ padding-top: 1.05rem; padding-bottom: 1.4rem; max-width: 1380px; }}
        [data-testid="stSidebar"] {{ background: {NAVY}; }}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: #F4F0EA; }}
        [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code {{
            color: {NAVY} !important; background: {SAND} !important;
        }}
        .brand-bar {{
            display: flex; align-items: center; justify-content: space-between;
            gap: 1rem; margin: 0 0 1.05rem 0; padding: 0.95rem 1.15rem;
            background: {NAVY}; border-radius: 18px;
            box-shadow: 0 12px 28px rgba(10,18,38,0.18);
        }}
        .brand-left {{ display: flex; align-items: center; gap: 0.85rem; }}
        .brand-mark {{
            width: 42px; height: 42px; border-radius: 10px; object-fit: cover;
            background: #000; box-shadow: 0 0 0 1px rgba(255,255,255,0.18);
        }}
        .brand-word {{
            font-family: "Cormorant Garamond", serif; font-size: 1.85rem;
            font-weight: 600; color: #fff; letter-spacing: 0.01em; line-height: 1;
        }}
        .brand-sub {{ color: rgba(255,255,255,0.72); font-size: 0.78rem; margin-top: 0.22rem; }}
        .brand-right {{ text-align: right; color: rgba(255,255,255,0.86); }}
        .brand-name {{ font-weight: 600; font-size: 0.92rem; color: #fff; }}
        .brand-meta {{ font-size: 0.74rem; color: rgba(255,255,255,0.68); margin-top: 0.12rem; }}
        .cat-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0 0 1rem 0; }}
        .cat-pill {{
            font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase;
            padding: 0.28rem 0.62rem; border-radius: 999px; background: #fff;
            color: {INK}; border: 1px solid rgba(24,46,99,0.12);
        }}
        .cat-pill.on {{ background: {ACCENT}; color: #fff; border-color: {ACCENT}; }}
        .panel {{
            background: #fff; border-radius: 18px; padding: 1rem 1.05rem 0.9rem;
            box-shadow: 0 10px 30px rgba(10,18,38,0.08); border: 1px solid rgba(24,46,99,0.06);
        }}
        .panel h3 {{
            font-family: "Cormorant Garamond", serif; font-size: 1.35rem;
            color: {NAVY}; margin: 0 0 0.75rem 0; font-weight: 600;
        }}
        .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.55rem; }}
        .metric-grid.two {{ grid-template-columns: 1fr 1fr; margin-top: 0.55rem; }}
        .metric {{
            background: {SAND}; border-radius: 12px; padding: 0.55rem 0.65rem 0.5rem;
        }}
        .metric .k {{ font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; color: {KTH}; font-weight: 600; }}
        .metric .v {{ font-size: 1.22rem; color: {NAVY}; font-weight: 600; margin-top: 0.15rem; line-height: 1.2; }}
        .metric.policy-clarify {{ background: #F3E0D2; }}
        .metric.policy-suggest {{ background: #DCE6FB; }}
        .empty {{ color: #5b6170; font-size: 0.92rem; padding: 1.4rem 0.2rem; }}
        .foot {{
            margin-top: 0.85rem; color: #6a6570; font-size: 0.75rem;
            display: flex; justify-content: space-between; gap: 1rem;
        }}
        [data-testid="stChatMessage"] {{ background: transparent; }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] li {{ color: {NAVY} !important; }}
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {{
            background: #fff; border: 1px solid rgba(24,46,99,0.10);
            border-radius: 14px; padding: 0.55rem 0.8rem; color: {NAVY};
        }}
        [data-testid="stBottomBlockContainer"] {{ background: {CREAM}; }}
        [data-testid="stChatInput"] {{ background: #fff; }}
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] p {{ color: {NAVY} !important; font-family: "Poppins", sans-serif; }}
        [data-testid="stExpander"] {{
            background: #fff; border-radius: 12px; border: 1px solid rgba(24,46,99,0.08);
            color: {NAVY};
        }}
        [data-testid="stExpander"] p, [data-testid="stExpander"] summary {{ color: {NAVY} !important; }}
        [data-testid="stCode"] {{ color: {NAVY} !important; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    icon = _b64(FIX_ICON) if FIX_ICON.is_file() else ""
    img = f'<img class="brand-mark" src="data:image/png;base64,{icon}" alt="Fixmeapp"/>' if icon else ""
    st.markdown(
        f"""
        <div class="brand-bar">
          <div class="brand-left">
            {img}
            <div>
              <div class="brand-word">Fixmeapp</div>
              <div class="brand-sub">Intent-aware booking · research prototype</div>
            </div>
          </div>
          <div class="brand-right">
            <div class="brand-name">Shuwei Song</div>
            <div class="brand-meta">KTH EECS · hosted by Fixmeapp AB</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _category_row(active: str | None) -> None:
    pills = []
    for cat in CATEGORIES:
        on = " on" if active == cat else ""
        pills.append(f'<span class="cat-pill{on}">{cat}</span>')
    st.markdown(f'<div class="cat-row">{"".join(pills)}</div>', unsafe_allow_html=True)


def _sidecar(last) -> None:
    if last is None:
        st.markdown(
            '<div class="panel"><h3>Sidecar</h3>'
            '<div class="empty">Type below. Intention stays local; the model realises the reply.</div></div>',
            unsafe_allow_html=True,
        )
        return
    pol_cls = "policy-clarify" if last.policy == "clarify" else "policy-suggest"
    st.markdown(
        f"""
        <div class="panel">
          <h3>Sidecar</h3>
          <div class="metric-grid">
            <div class="metric {pol_cls}"><div class="k">Policy</div><div class="v">{last.policy}</div></div>
            <div class="metric"><div class="k">M1</div><div class="v">{last.m1_category}</div></div>
            <div class="metric"><div class="k">H_cat</div><div class="v">{last.category_entropy:.2f}</div></div>
          </div>
          <div class="metric-grid two">
            <div class="metric"><div class="k">Category mass</div><div class="v">{last.top1_category_prob:.2f}</div></div>
            <div class="metric"><div class="k">Entropy H(b)</div><div class="v">{last.posterior_entropy:.2f}</div></div>
          </div>
          <div class="foot"><span>Belief top class · {last.pred_category}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _prompt_for_display(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not ln.startswith("## Intention context")]
    return "\n".join(lines).lstrip()


def _init() -> None:
    if "runner" not in st.session_state:
        st.session_state.runner = DefenceDiningRunner()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _reset() -> None:
    st.session_state.runner = DefenceDiningRunner()
    st.session_state.messages = []


def _submit(text: str, enable_gpt: bool, model: str) -> None:
    runner: DefenceDiningRunner = st.session_state.runner
    turn = runner.step_text(text)
    reply = surface_reply(
        turn,
        st.session_state.messages,
        enable_gpt=enable_gpt,
        model=model,
    )
    st.session_state.messages.append(ChatMessage(role="user", content=turn.text))
    st.session_state.messages.append(
        ChatMessage(
            role="assistant",
            content=reply,
            turn_type=DialogueTurnType.CLARIFY if turn.policy == "clarify" else DialogueTurnType.TASK,
            pipeline_result=turn.result,
            intention_block=turn.block,
            surface_policy=turn.policy,
        )
    )


def main() -> None:
    load_project_env()
    st.set_page_config(
        page_title="Fixmeapp · Shuwei Song",
        page_icon=str(FIX_ICON) if FIX_ICON.is_file() else None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _css()
    _init()

    runner: DefenceDiningRunner = st.session_state.runner
    cfg = runner.config
    gpt = gpt_connection_status(model=cfg.dialogue.model, enable_gpt=cfg.dialogue.enable_gpt)
    lines = runner.lines_to_type()
    last = runner.history[-1] if runner.history else None
    active = last.pred_category if last else None

    with st.sidebar:
        st.markdown("**Prompter**")
        st.caption("Collapse this panel before you record.")
        for i, line in enumerate(lines, 1):
            st.markdown(f"**Turn {i}** — type exactly")
            st.code(line, language=None)
            if i <= len(BEATS):
                st.caption(BEATS[i - 1])
        if st.button("Reset session", use_container_width=True):
            _reset()
            st.rerun()
        st.caption("Shuwei Song · off-camera notes")
        if gpt["connected"]:
            st.caption(f"Surface: GPT ({gpt['model']})")
        else:
            st.caption("Surface: local fallback — not scored")

    _header()
    _category_row(active)

    chat_col, side_col = st.columns([3, 2], gap="large")
    user_av = str(SS_ICON) if SS_ICON.is_file() else None
    bot_av = str(FIX_ICON) if FIX_ICON.is_file() else None

    with chat_col:
        if not st.session_state.messages:
            st.caption("Type the first message. Intention stays local; the model realises the reply.")
        for msg in st.session_state.messages:
            avatar = user_av if msg.role == "user" else bot_av
            with st.chat_message(msg.role, avatar=avatar):
                st.markdown(msg.content)
        typed = st.chat_input("Tell me what you need…")
        if typed:
            _submit(typed, bool(gpt["connected"]), cfg.dialogue.model)
            st.rerun()

    with side_col:
        _sidecar(last)
        if last is not None:
            with st.expander("Intention block → LLM", expanded=False):
                st.code(_prompt_for_display(last.prompt_text), language="markdown")
                st.markdown("**Withheld from the model**")
                for item in last.withheld_from_llm:
                    st.markdown(f"- {item}")

    st.markdown(
        '<div class="foot"><span>Fixmeapp AB · Stockholm</span>'
        '<span>Master’s thesis prototype · Shuwei Song, KTH EECS, 2026</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
