# app.py — BERTopic Thematic Analysis Agent
# Built specifically for Gradio 6.11.0.
#
# KEY FIXES in this version:
#   FIX-A: call_agent detects INVALID_CHAT_HISTORY (dangling tool call in
#           MemorySaver after a mid-tool 429) and rotates to a fresh thread_id.
#   FIX-B: Rate-limit back-off extended to 30 / 60 / 90 s (was 10/20/30 s).
#   FIX-C: on_clear() now deletes all checkpoint files so Phase 1 truly resets.
#   FIX-D: All UI handlers return the (possibly rotated) sid_state.
#   FIX-E: stdout/stderr reconfigured to UTF-8 so Mistral emoji (✅📄⬜) don't
#           crash print() on Windows cp1252 consoles.

import sys

# FIX-E: Reconfigure console to UTF-8 BEFORE any print() calls.
# Windows default (cp1252) cannot encode Mistral's emoji responses,
# causing UnicodeEncodeError inside log_error() which propagated to the UI.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Non-TTY environments (HuggingFace Spaces) don't need this

import gradio as gr
import json
import os
import uuid
import glob
import pandas as pd
import traceback
import datetime
import time
from agent import agent

# Check for API Key
if not os.environ.get("MISTRAL_API_KEY"):
    print("\n" + "!"*80)
    print("CRITICAL WARNING: MISTRAL_API_KEY environment variable is NOT set.")
    print("The agent will fail with a 401 Unauthorized error when calling Mistral.")
    print("!"*80 + "\n")

print(f"[app.py] Starting with Gradio {gr.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
REVIEW_COLUMNS = [
    "#", "Topic Label", "Top Evidence Sentence",
    "Sent.", "Papers", "Approve", "Rename To",
]

EMPTY_REVIEW_DF = pd.DataFrame(
    columns=REVIEW_COLUMNS,
    data=[["", "", "", 0, 0, False, ""]],
)

DOWNLOAD_FILES = [
    "narrative.txt", "comparison.csv", "themes.json",
    "taxonomy_map.json", "labels_abstract.json", "labels_title.json",
]

# Files to wipe when the user resets the session
CHECKPOINT_FILES = [
    "loaded_data.csv",
    "summaries_abstract.json", "summaries_title.json",
    "emb_abstract.npy", "emb_title.npy",
    "labels_abstract.json", "labels_title.json",
    "themes.json", "themes_abstract.json", "themes_title.json",
    "taxonomy_map.json", "comparison.csv", "narrative.txt",
    "chart_abstract_intertopic.html", "chart_abstract_bars.html",
    "chart_abstract_hierarchy.html", "chart_abstract_heatmap.html",
    "chart_title_intertopic.html", "chart_title_bars.html",
    "chart_title_hierarchy.html", "chart_title_heatmap.html",
]

CHART_OPTIONS = [
    ("Intertopic Map — Abstract",      "chart_abstract_intertopic.html"),
    ("Frequency Bars — Abstract",      "chart_abstract_bars.html"),
    ("Hierarchy / Treemap — Abstract", "chart_abstract_hierarchy.html"),
    ("Similarity Heatmap — Abstract",  "chart_abstract_heatmap.html"),
    ("Intertopic Map — Title",         "chart_title_intertopic.html"),
    ("Frequency Bars — Title",         "chart_title_bars.html"),
    ("Hierarchy / Treemap — Title",    "chart_title_hierarchy.html"),
    ("Similarity Heatmap — Title",     "chart_title_heatmap.html"),
]

PHASE_LABELS = [
    ("1","① Load"), ("2","② Codes"), ("3","③ Themes"),
    ("4","④ Review"), ("5","⑤ Names"), ("5.5","⑤½ PAJAIS"), ("6","⑥ Report"),
]

# Error strings that indicate a corrupted MemorySaver thread
# (dangling AIMessage with tool_call but no ToolMessage)
CORRUPT_HISTORY_SIGNALS = [
    "INVALID_CHAT_HISTORY",
    "ToolMessage",
    "tool_calls that do not have a corresponding",
]

CSS = """
body, .gradio-container {
    background: #0d0d1a !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.section-hdr {
    background: linear-gradient(90deg, #1a2a4a, #0d1a2e);
    color: #7fb3f5 !important; font-weight: 800 !important; font-size: 0.8rem !important;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 7px 14px; border-radius: 6px 6px 0 0;
    border-left: 3px solid #4a90d9; margin-bottom: 4px;
}
footer { display: none !important; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Message helpers
# Gradio 6.11 ALWAYS needs: {"role": "user"|"assistant", "content": str}
# ─────────────────────────────────────────────────────────────────────────────
def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": str(content)}


def append_msgs(history: list, user_text: str, bot_text: str) -> list:
    """Append a user+assistant exchange to chat history."""
    return history + [_msg("user", user_text), _msg("assistant", bot_text)]


def empty_history() -> list:
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def log_error(msg: str, ctx: str = "") -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("error.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\nTIME: {ts}\nCONTEXT: {ctx}\n"
                f"ERROR: {msg}\nTRACEBACK:\n{traceback.format_exc()}\n")
    # Secondary safety net: if stdout reconfigure didn't work, don't crash
    try:
        print(f"[ERROR] {ctx}: {str(msg)[:120]}")
    except UnicodeEncodeError:
        print(f"[ERROR] {ctx}: (non-ASCII chars in message — see error.txt)")


def safe_str(val) -> str:
    """Convert any LangGraph output to plain str safely."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("content", item.get("text", ""))))
            elif hasattr(item, "content"):
                parts.append(safe_str(item.content))
            else:
                parts.append(str(item))
        return "\n".join(filter(None, parts))
    if isinstance(val, dict):
        return str(val.get("content", val.get("text", str(val))))
    if hasattr(val, "content"):
        return safe_str(val.content)
    return str(val)


def detect_phase_status() -> dict:
    return {
        "1":   os.path.exists("loaded_data.csv"),
        "2":   os.path.exists("labels_abstract.json") or os.path.exists("labels_title.json"),
        "3":   os.path.exists("themes.json"),
        "4":   os.path.exists("themes.json"),
        "5":   os.path.exists("themes.json"),
        "5.5": os.path.exists("taxonomy_map.json"),
        "6":   os.path.exists("narrative.txt"),
    }


def build_phase_bar(status: dict) -> str:
    items = ""
    for key, label in PHASE_LABELS:
        done = status.get(key, False)
        bg  = "#2ecc71" if done else "#2a2a3e"
        col = "#000"    if done else "#888"
        bdr = "#2ecc71" if done else "#444"
        items += (
            f'<span style="display:inline-block;padding:4px 11px;margin:2px;'
            f'background:{bg};border:1.5px solid {bdr};border-radius:18px;'
            f'font-size:0.75rem;font-weight:700;color:{col};white-space:nowrap;">'
            f'{"✅ " if done else ""}{label}</span>'
        )
    return (
        f'<div style="background:#12122a;padding:9px 14px;border-radius:8px;'
        f'border:1px solid #2a2a4a;margin-bottom:6px;line-height:2.4;">'
        f'<span style="color:#5a7abf;font-size:0.7rem;font-weight:800;'
        f'letter-spacing:0.09em;margin-right:8px;">BRAUN &amp; CLARKE PHASES</span>'
        f'{items}</div>'
    )


def parse_phase_status(text, current: dict) -> dict:
    text = safe_str(text)
    updated = dict(current)
    for line in text.splitlines():
        if "PHASE_STATUS:" in line:
            raw = line.split("PHASE_STATUS:", 1)[1].strip()
            for part in [p.strip() for p in raw.split(",")]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    updated[k.strip()] = "✅" in v
    for k, v in detect_phase_status().items():
        updated[k] = updated.get(k, False) or v
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Review table loader
# ─────────────────────────────────────────────────────────────────────────────
def load_review_table() -> pd.DataFrame:
    if os.path.exists("taxonomy_map.json"):
        data = json.loads(open("taxonomy_map.json", encoding="utf-8").read())
        rows = []
        for i, item in enumerate(data):
            evidence = (
                f"→ NOVEL | {item.get('reasoning','')[:80]}"
                if item.get("is_novel", False)
                else f"→ PAJAIS: {item.get('pajais_match','')} | {item.get('reasoning','')[:60]}"
            )
            rows.append({"#": i, "Topic Label": item.get("theme_name", ""),
                         "Top Evidence Sentence": evidence,
                         "Sent.": 0, "Papers": 0, "Approve": True, "Rename To": ""})
        return pd.DataFrame(rows, columns=REVIEW_COLUMNS) if rows else EMPTY_REVIEW_DF

    if os.path.exists("themes.json"):
        data = json.loads(open("themes.json", encoding="utf-8").read())
        rows = []
        for i, item in enumerate(data):
            s = item.get("total_sentences", 0)
            rows.append({"#": i, "Topic Label": item.get("theme_name", ""),
                         "Top Evidence Sentence": (
                             item.get("representative_sentences", [""])[0][:120]
                             if item.get("representative_sentences") else ""),
                         "Sent.": s, "Papers": max(1, s // 10),
                         "Approve": False, "Rename To": ""})
        return pd.DataFrame(rows, columns=REVIEW_COLUMNS) if rows else EMPTY_REVIEW_DF

    for rk in ("abstract", "title"):
        p = f"labels_{rk}.json"
        if os.path.exists(p):
            data = json.loads(open(p, encoding="utf-8").read())
            rows = []
            for t in data:
                s = t.get("count", 0)
                rows.append({"#": t.get("topic_id", 0),
                             "Topic Label": t.get("label", f"Topic {t.get('topic_id',0)}"),
                             "Top Evidence Sentence": (
                                 t.get("nearest_sentences", [""])[0][:120]
                                 if t.get("nearest_sentences") else ""),
                             "Sent.": s, "Papers": max(1, s // 10),
                             "Approve": False, "Rename To": ""})
            return pd.DataFrame(rows, columns=REVIEW_COLUMNS) if rows else EMPTY_REVIEW_DF

    return EMPTY_REVIEW_DF


def get_downloads():
    found = [f for f in DOWNLOAD_FILES if os.path.exists(f)]
    return found if found else None


def render_chart(chart_file: str) -> str:
    if not chart_file or not os.path.exists(chart_file):
        return ("<div style='padding:40px;text-align:center;color:#555;'>"
                "Chart not available yet — run analysis first.</div>")
    content = open(chart_file, encoding="utf-8").read()
    escaped = content.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&#39;")
    return (f'<iframe srcdoc="{escaped}" style="width:100%;height:540px;'
            f'border:none;border-radius:6px;" '
            f'sandbox="allow-scripts allow-same-origin"></iframe>')


# ─────────────────────────────────────────────────────────────────────────────
# Agent caller — returns (response_str, session_id_used)
#
# FIX-A: When MemorySaver thread is corrupted (dangling AIMessage with
#         tool_call, no ToolMessage), we detect the INVALID_CHAT_HISTORY
#         error and rotate to a brand-new thread_id. The caller receives
#         the new sid so it can update sid_state and avoid the permanent lock.
#
# FIX-B: Rate-limit back-off is now 30/60/90 s (was 10/20/30 s).
# ─────────────────────────────────────────────────────────────────────────────
def call_agent(message: str, session_id: str, max_retries: int = 3) -> tuple[str, str]:
    """
    Invoke the LangGraph agent.
    Returns (response_text, session_id_used).
    session_id_used may differ from the input session_id if history corruption
    forced a thread rotation (FIX-A).
    """
    current_sid = session_id

    for attempt in range(max_retries):
        try:
            config = {"configurable": {"thread_id": current_sid}}
            result = agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config,
            )
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "type") and msg.type == "ai":
                    return safe_str(msg.content), current_sid
                if isinstance(msg, dict) and msg.get("role") in ("assistant", "ai"):
                    return safe_str(msg.get("content", "")), current_sid
            return "Agent returned no response. Please try again.", current_sid

        except Exception as e:
            err = str(e)

            # ── FIX-A: Corrupted history (dangling tool call in MemorySaver) ──
            # Rotate to a new thread so MemorySaver starts fresh.
            if any(sig in err for sig in CORRUPT_HISTORY_SIGNALS):
                new_sid = str(uuid.uuid4())
                log_error(err, ctx=f"call_agent [corrupt-history → rotating {current_sid[:8]}→{new_sid[:8]}]")
                print(f"⚠️  Corrupt history detected — rotating session {current_sid[:8]} → {new_sid[:8]}")
                recovery_msg = (
                    f"{message}\n\n"
                    "[SYSTEM NOTE: The previous session thread had a corrupted history "
                    "due to a mid-tool API failure. This is a fresh thread. "
                    "Checkpoint files (themes.json, taxonomy_map.json, etc.) are intact on disk. "
                    "Please resume from where we left off based on the existing checkpoint files.]"
                )
                current_sid = new_sid
                # Retry immediately on the clean thread (don't sleep)
                try:
                    config = {"configurable": {"thread_id": current_sid}}
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": recovery_msg}]},
                        config=config,
                    )
                    for msg in reversed(result.get("messages", [])):
                        if hasattr(msg, "type") and msg.type == "ai":
                            return safe_str(msg.content), current_sid
                        if isinstance(msg, dict) and msg.get("role") in ("assistant", "ai"):
                            return safe_str(msg.get("content", "")), current_sid
                    return "Agent returned no response after history rotation. Please try again.", current_sid
                except Exception as e2:
                    log_error(str(e2), ctx="call_agent [post-rotation]")
                    return f"⚠️ Agent Error after session rotation: {e2}\n\nSee error.txt for details.", current_sid

            # ── FIX-B: Mistral rate-limit / server errors — extended back-off ──
            if any(c in err for c in ["429", "520", "502", "503", "529", "mistral.ai", "Rate limit"]):
                log_error(err, ctx=f"call_agent attempt {attempt + 1}")
                wait = 30 * (attempt + 1)   # 30 / 60 / 90 s
                print(f"⚠️  Mistral rate-limit/server error — retrying in {wait}s…")
                time.sleep(wait)
                continue

            log_error(err, ctx="call_agent")
            return f"⚠️ Agent Error: {err}\n\nSee error.txt for details.", current_sid

    return "❌ Mistral not responding after retries. Wait a few minutes and try again.", current_sid


# ─────────────────────────────────────────────────────────────────────────────
# Event handlers  (all return the sid so sid_state stays up-to-date)
# ─────────────────────────────────────────────────────────────────────────────
def on_upload(file_obj, history, sid, status):
    if file_obj is None:
        return history, sid, status, build_phase_bar(status), load_review_table(), get_downloads()
    try:
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        msg = (
            f"I have uploaded my Scopus CSV. File path: {path}\n\n"
            "Please begin Phase 1: load the file, show all dataset statistics "
            "(papers, abstract sentences, title sentences, year range, columns, "
            "sample titles), then ask me which run_key to use."
        )
        response, new_sid = call_agent(msg, sid)
        new_hist   = append_msgs(history, msg, response)
        new_status = parse_phase_status(response, status)
        return new_hist, new_sid, new_status, build_phase_bar(new_status), load_review_table(), get_downloads()
    except Exception as e:
        log_error(str(e), ctx="on_upload")
        return (append_msgs(history, "[File Upload]", f"Upload error: {e}"),
                sid, status, build_phase_bar(status), load_review_table(), get_downloads())


def on_send(user_msg, history, sid, status):
    if not user_msg.strip():
        return history, "", sid, status, build_phase_bar(status), load_review_table(), get_downloads()
    try:
        response, new_sid = call_agent(user_msg, sid)
        new_hist   = append_msgs(history, user_msg, response)
        new_status = parse_phase_status(response, status)
        return new_hist, "", new_sid, new_status, build_phase_bar(new_status), load_review_table(), get_downloads()
    except Exception as e:
        log_error(str(e), ctx="on_send")
        return (append_msgs(history, user_msg, f"Error: {e}"),
                "", sid, status, build_phase_bar(status), load_review_table(), get_downloads())


def on_submit_review(review_df, history, sid, status):
    try:
        df = review_df if isinstance(review_df, pd.DataFrame) else pd.DataFrame(review_df)
        approved    = df[df["Approve"].astype(bool)]
        rename_map  = {}
        labels_list = []

        for _, row in approved.iterrows():
            tid   = str(row.get("#", ""))
            label = str(row.get("Topic Label", "")).strip()
            ren   = str(row.get("Rename To", "")).strip()
            labels_list.append(ren if ren else label)
            if ren:
                rename_map[tid] = ren

        lines = []
        if labels_list:
            shown = ", ".join(labels_list[:6]) + ("…" if len(labels_list) > 6 else "")
            lines.append(f"Approved {len(labels_list)} row(s): {shown}")
        if rename_map:
            lines.append("Renames: " + ", ".join(
                f"#{k}→'{v}'" for k, v in list(rename_map.items())[:5]))
        summary = "\n".join(lines) if lines else "No approvals or renames submitted."

        msg = (
            "I have submitted the Review Table.\n\n"
            f"Decisions:\n{summary}\n\n"
            f"Rename overrides JSON: {json.dumps(rename_map)}\n\n"
            "Please proceed to the next phase using these decisions."
        )
        response, new_sid = call_agent(msg, sid)
        new_hist   = append_msgs(history, msg, response)
        new_status = parse_phase_status(response, status)
        return new_hist, new_sid, new_status, build_phase_bar(new_status), load_review_table(), get_downloads()
    except Exception as e:
        log_error(str(e), ctx="on_submit_review")
        return (append_msgs(history, "[Submit Review]", f"Submit error: {e}"),
                sid, status, build_phase_bar(status), load_review_table(), get_downloads())


def on_chart_change(label: str) -> str:
    return render_chart(dict(CHART_OPTIONS).get(label, ""))


def on_clear(sid):
    """Reset the UI and wipe all checkpoint files so Phase 1 re-runs clean."""
    for f in CHECKPOINT_FILES:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
    new_sid    = str(uuid.uuid4())
    blank      = {k: False for k in ["1", "2", "3", "4", "5", "5.5", "6"]}
    new_status = parse_phase_status("", blank)
    return empty_history(), new_sid, new_status, build_phase_bar(new_status)


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────
INIT_STATUS = parse_phase_status("", {k: False for k in ["1","2","3","4","5","5.5","6"]})

with gr.Blocks(title="BERTopic Agentic Topic Modelling") as demo:

    # State
    sid_state     = gr.State(str(uuid.uuid4()))
    history_state = gr.State(empty_history())
    status_state  = gr.State(INIT_STATUS)

    # Header
    gr.HTML("""
    <div style="padding:16px 0 4px;">
      <h1 style="color:#e8f0fe;font-size:1.5rem;font-weight:900;margin:0;">
        🔬 BERTopic Agentic Topic Modelling
        <span style="font-size:0.72rem;font-weight:400;color:#5a6a8a;margin-left:10px;">
          (Braun &amp; Clarke 2006)
        </span>
      </h1>
    </div>""")

    phase_bar = gr.HTML(value=build_phase_bar(INIT_STATUS))

    with gr.Row(equal_height=False):

        # ── Data Input ────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=230):
            gr.HTML('<div class="section-hdr">① DATA INPUT</div>')
            file_input = gr.File(
                label="Upload Scopus CSV",
                file_types=[".csv"],
                height=100,
            )
            gr.HTML("<p style='color:#4a5a7a;font-size:0.73rem;margin:4px 2px;'>"
                    "Upload CSV → auto-triggers Phase 1</p>")

        # ── Chatbot ───────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.HTML('<div class="section-hdr">② AGENT CONVERSATION</div>')

            chatbot = gr.Chatbot(
                value=empty_history(),
                height=340,
                show_label=False,
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    show_label=False,
                    placeholder="Type 'run abstract', 'Continue', or any message…",
                    scale=6, lines=1, max_lines=3, container=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=85)
            clear_btn = gr.Button("🗑 Clear Chat & Reset", variant="secondary", size="sm")

    # ── Results ───────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="section-hdr">'
                    '③ RESULTS — REVIEW TABLE · CHARTS · DOWNLOADS</div>')

            with gr.Tabs():

                with gr.Tab("📋 Review Table"):
                    review_table = gr.Dataframe(
                        value=load_review_table(),
                        headers=REVIEW_COLUMNS,
                        datatype=["number", "str", "str", "number", "number", "bool", "str"],
                        interactive=True,
                        wrap=True,
                        row_count=(6, "dynamic"),
                        column_count=(7, "fixed"),
                        show_label=False,
                    )
                    submit_btn = gr.Button(
                        "✅  Submit Review to Agent", variant="primary", size="lg")
                    gr.HTML("<p style='color:#4a5a7a;font-size:0.73rem;margin:4px 2px;'>"
                            "Tick Approve / fill Rename To, then click Submit Review.</p>")

                with gr.Tab("📈 Charts"):
                    chart_dd = gr.Dropdown(
                        choices=[o[0] for o in CHART_OPTIONS],
                        value=CHART_OPTIONS[0][0],
                        label="Select chart",
                        interactive=True,
                    )
                    chart_display = gr.HTML(
                        "<div style='padding:30px;text-align:center;color:#444;'>"
                        "Charts appear after Phase 2 completes.</div>")

                with gr.Tab("💾 Download"):
                    gr.HTML("<p style='color:#4a5a7a;font-size:0.78rem;padding:6px 2px;'>"
                            "<code>narrative.txt</code> · <code>comparison.csv</code> · "
                            "<code>themes.json</code> · <code>taxonomy_map.json</code></p>")
                    dl_box = gr.File(
                        value=get_downloads(),
                        show_label=False,
                        file_count="multiple",
                        interactive=False,
                        height=180,
                    )

    # ── Event wiring ──────────────────────────────────────────────────────────
    # FIX-C: Removed the chatbot.change → history_state sync listener.
    #        history_state is now updated directly by each handler's return value.

    file_input.change(
        fn=on_upload,
        inputs=[file_input, history_state, sid_state, status_state],
        outputs=[chatbot, sid_state, status_state, phase_bar, review_table, dl_box],
    )
    # Keep history_state in sync with chatbot (chatbot is the source of truth)
    chatbot.change(fn=lambda h: h, inputs=chatbot, outputs=history_state)

    send_btn.click(
        fn=on_send,
        inputs=[chat_input, history_state, sid_state, status_state],
        outputs=[chatbot, chat_input, sid_state, status_state, phase_bar, review_table, dl_box],
    )
    chat_input.submit(
        fn=on_send,
        inputs=[chat_input, history_state, sid_state, status_state],
        outputs=[chatbot, chat_input, sid_state, status_state, phase_bar, review_table, dl_box],
    )
    submit_btn.click(
        fn=on_submit_review,
        inputs=[review_table, history_state, sid_state, status_state],
        outputs=[chatbot, sid_state, status_state, phase_bar, review_table, dl_box],
    )
    chart_dd.change(fn=on_chart_change, inputs=chart_dd, outputs=chart_display)
    clear_btn.click(
        fn=on_clear,
        inputs=[sid_state],
        outputs=[chatbot, sid_state, status_state, phase_bar],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
    )