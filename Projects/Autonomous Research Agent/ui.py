# ui.py
import time
import threading
import gradio as gr
from core.graph import build_graph
from core.llm import validate_api_key
from dotenv import load_dotenv

load_dotenv()
validate_api_key()

graph = build_graph()


def run_research_ui(topic: str, depth: str, progress=gr.Progress()):
    """Run research and stream progress to UI"""
    if not topic.strip():
        return "⚠️ Please enter a research topic", "", "", ""

    config = {
        "configurable": {
            "thread_id": f"ui_{int(time.time())}"
        }
    }

    progress(0, desc="Initializing agents...")
    start = time.time()
    log_lines = []

    agent_progress = {
        "researcher": 0.25,
        "analyst":    0.50,
        "writer":     0.70,
        "critic":     0.85,
        "finalizer":  0.95,
        "supervisor": None
    }

    try:
        for step in graph.stream(
            {
                "topic":        topic,
                "depth":        depth,
                "messages":     [],
                "raw_sources":  [],
                "key_insights": [],
                "completed":    [],
                "iteration":    0,
                "start_time":   start
            },
            config=config,
            stream_mode="updates"
        ):
            for node, update in step.items():
                elapsed = time.time() - start
                pct = agent_progress.get(node)

                msg = f"[{elapsed:.0f}s] {node.upper()}"
                if update.get("status"):
                    msg += f" → {update['status']}"
                if update.get("quality_score"):
                    msg += f" | Score: {update['quality_score']}/10"

                log_lines.append(msg)

                if pct:
                    progress(pct, desc=f"Running {node}...")

        progress(1.0, desc="Done!")

        # Get final state
        state = graph.get_state(config).values
        report      = state.get("final_report", "No report generated")
        score       = state.get("quality_score", "N/A")
        word_count  = state.get("word_count", 0)
        sources     = state.get("raw_sources", [])
        insights    = state.get("key_insights", [])
        elapsed     = time.time() - start

        stats = f"""**Research Complete** ✅
- ⏱️ Time: {elapsed:.0f}s
- ⭐ Quality: {score}/10
- 📝 Words: {word_count}
- 🔗 Sources: {len(sources)}
- 💡 Insights: {len(insights)}"""

        sources_md = "\n".join(f"- {s}" for s in sources[:8]) or "None found"
        log = "\n".join(log_lines)

        return report, stats, sources_md, log

    except Exception as e:
        return f"❌ Error: {e}", "", "", str(e)


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(
    title="Autonomous Research Agent"
) as demo:

    gr.Markdown("""
    # 🔬 Autonomous Research Agent
    **Multi-agent AI system that researches any topic autonomously**
    Researcher → Analyst → Writer → Critic → Final Report
    """)

    with gr.Row():
        with gr.Column(scale=2):
            topic_input = gr.Textbox(
                label="Research Topic",
                placeholder="e.g. Impact of AI agents on software development in 2025",
                lines=2
            )
            with gr.Row():
                depth_radio = gr.Radio(
                    choices=["quick", "standard", "deep"],
                    value="standard",
                    label="Research Depth"
                )
                run_btn = gr.Button(
                    "🚀 Start Research",
                    variant="primary",
                    scale=1
                )

        with gr.Column(scale=1):
            stats_display = gr.Markdown("*Research stats will appear here*")
            sources_display = gr.Markdown(
                label="Sources",
                value="*Sources will appear here*"
            )

    with gr.Tabs():
        with gr.Tab("📄 Final Report"):
            report_output = gr.Markdown(
                value="*Your research report will appear here*",
                height=500
            )

        with gr.Tab("📋 Agent Log"):
            log_output = gr.Textbox(
                label="Execution Log",
                lines=20,
                interactive=False
            )

    gr.Examples(
        examples=[
            ["How LangGraph is used in production AI systems", "standard"],
            ["The future of open source LLMs in 2025",        "quick"],
            ["RAG vs fine-tuning: which approach wins",       "standard"],
            ["Multi-agent AI systems in enterprise software",  "deep"],
        ],
        inputs=[topic_input, depth_radio],
        label="Example Research Topics"
    )

    run_btn.click(
        fn=run_research_ui,
        inputs=[topic_input, depth_radio],
        outputs=[report_output, stats_display, sources_display, log_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        inbrowser=True,
        share=False
    )