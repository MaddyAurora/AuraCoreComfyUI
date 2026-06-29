"""
AuraCoreComfyUI - Gradio web frontend for ComfyUI

Each tab = one AI model.
Inside each tab, a dropdown selects which workflow JSON to run.
Workflows live in: ./workflows/<model_folder>/<workflow>.json

ComfyUI must be running separately (default: http://127.0.0.1:8188)
"""

import json
import logging
import os
import random
import time
import urllib.request
import urllib.parse
import uuid
import websocket
from pathlib import Path
from typing import Any

import gradio as gr

# ───────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_URL  = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
WS_URL       = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws"

APP_DIR       = Path(__file__).parent
WORKFLOWS_DIR = APP_DIR / "workflows"
OUTPUTS_DIR   = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("AuraCoreComfyUI")

# ───────────────────────────────────────────────
# Model tab definitions
# Each entry: (Tab label, subfolder name inside workflows/)
# ───────────────────────────────────────────────
MODEL_TABS = [
    ("Krea 2",       "krea2"),
    ("Qwen Image",   "qwen_image"),
    ("Ideogram 4",   "ideogram4"),
    ("Klein 9B",     "klein9b"),
]


# ───────────────────────────────────────────────
# ComfyUI API helpers
# ───────────────────────────────────────────────
CLIENT_ID = str(uuid.uuid4())


def comfy_api(endpoint: str, data: dict | None = None) -> Any:
    """GET (data=None) or POST to ComfyUI REST API."""
    url = f"{COMFYUI_URL}/{endpoint}"
    if data is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        log.error(f"ComfyUI API error [{endpoint}]: {e}")
        return None


def check_connection() -> tuple[bool, str]:
    result = comfy_api("system_stats")
    if result:
        return True, f"✅ Connected — {COMFYUI_URL}"
    return False, f"❌ Not connected — {COMFYUI_URL}"


def queue_prompt(workflow: dict) -> str | None:
    payload = {"prompt": workflow, "client_id": CLIENT_ID}
    result = comfy_api("prompt", payload)
    if result and "prompt_id" in result:
        return result["prompt_id"]
    return None


def wait_for_result(prompt_id: str, timeout: int = 300) -> list[Path]:
    ws = websocket.WebSocket()
    try:
        ws.connect(f"{WS_URL}?clientId={CLIENT_ID}")
    except Exception as e:
        log.error(f"WebSocket connect failed: {e}")
        return []

    start = time.time()
    try:
        while time.time() - start < timeout:
            try:
                raw = ws.recv()
                msg = json.loads(raw) if isinstance(raw, str) else {}
            except Exception:
                break
            mtype = msg.get("type", "")
            data  = msg.get("data", {})
            if mtype == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
                break
    finally:
        ws.close()

    history = comfy_api(f"history/{prompt_id}")
    if not history or prompt_id not in history:
        return []

    saved = []
    for node_id, node_out in history[prompt_id].get("outputs", {}).items():
        for img in node_out.get("images", []):
            fname     = img["filename"]
            subfolder = img.get("subfolder", "")
            folder    = img.get("type", "output")
            params    = urllib.parse.urlencode({"filename": fname, "subfolder": subfolder, "type": folder})
            dest      = OUTPUTS_DIR / fname
            try:
                urllib.request.urlretrieve(f"{COMFYUI_URL}/view?{params}", dest)
                saved.append(dest)
            except Exception as e:
                log.error(f"Failed to save image {fname}: {e}")
    return saved


# ───────────────────────────────────────────────
# Workflow helpers
# ───────────────────────────────────────────────

def get_workflows_for_model(subfolder: str) -> list[Path]:
    folder = WORKFLOWS_DIR / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    return sorted(folder.glob("*.json"))


def workflow_choices(subfolder: str) -> list[str]:
    files = get_workflows_for_model(subfolder)
    if not files:
        return ["(no workflows yet)"]
    return [f.stem.replace("_", " ").replace("-", " ").title() for f in files]


def workflow_path_from_choice(subfolder: str, choice: str) -> Path | None:
    files = get_workflows_for_model(subfolder)
    for f in files:
        if f.stem.replace("_", " ").replace("-", " ").title() == choice:
            return f
    return None


def load_workflow(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inject_text2img_params(
    workflow: dict,
    positive: str,
    negative: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
) -> dict:
    wf = json.loads(json.dumps(workflow))

    for node_id, node in wf.items():
        ctype = node.get("class_type", "")
        if ctype in ("KSampler", "KSamplerAdvanced"):
            inp = node.get("inputs", {})
            inp["steps"]   = steps
            inp["cfg"]     = cfg
            inp["seed"]    = seed if seed != -1 else random.randint(0, 2**31)
            inp["denoise"] = inp.get("denoise", 1.0)
        if ctype in ("EmptyLatentImage", "EmptySD3LatentImage"):
            inp = node.get("inputs", {})
            inp["width"]  = width
            inp["height"] = height

    clip_nodes = [(nid, n) for nid, n in wf.items()
                  if n.get("class_type") == "CLIPTextEncode"]
    if len(clip_nodes) >= 1:
        clip_nodes[0][1]["inputs"]["text"] = positive
    if len(clip_nodes) >= 2:
        clip_nodes[1][1]["inputs"]["text"] = negative

    return wf


# ───────────────────────────────────────────────
# Tab builder
# ───────────────────────────────────────────────

def build_model_tab(label: str, subfolder: str):
    choices = workflow_choices(subfolder)

    with gr.TabItem(label):
        # Top bar: dropdown (no label) + Refresh + Generate stacked on the right
        with gr.Row(equal_height=True):
            workflow_dd = gr.Dropdown(
                choices=choices,
                value=choices[0],
                label="",          # no label — removes the "Workflow" heading
                show_label=False,
                scale=4,
                interactive=True,
            )
            with gr.Column(scale=1, min_width=110):
                reload_btn   = gr.Button("🔄 Refresh",  size="sm")
                generate_btn = gr.Button("🎨 Generate", variant="primary", size="sm")

        # Main content row
        with gr.Row():
            with gr.Column(scale=2):
                pos_prompt = gr.Textbox(
                    label="Positive Prompt",
                    placeholder="Describe what you want to generate...",
                    lines=4,
                )
                neg_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, watermark...",
                    lines=2,
                )
                with gr.Row():
                    steps = gr.Slider(1, 60,    value=20,  step=1,   label="Steps")
                    cfg   = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFG")
                with gr.Row():
                    width  = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                with gr.Row():
                    seed          = gr.Number(value=-1, label="Seed  (−1 = random)", precision=0)
                    rand_seed_btn = gr.Button("🎲 Randomize", size="sm")
                status_box = gr.Textbox(label="Status", interactive=False, lines=1)

            with gr.Column(scale=3):
                output_gallery = gr.Gallery(
                    label="Output",
                    columns=2,
                    preview=True,
                    height=620,
                )

        # Reload workflow list without restarting
        def reload_workflows():
            new_choices = workflow_choices(subfolder)
            return gr.Dropdown(choices=new_choices, value=new_choices[0])

        reload_btn.click(fn=reload_workflows, outputs=workflow_dd)

        # Randomize seed
        rand_seed_btn.click(fn=lambda: random.randint(0, 2**31), outputs=seed)

        # Generate
        def run_generation(wf_choice, pos, neg, st, cf, w, h, sd):
            ok, msg = check_connection()
            if not ok:
                yield [], msg
                return

            if wf_choice == "(no workflows yet)":
                yield [], f"❌ No workflow selected. Add a JSON to workflows/{subfolder}/"
                return

            wf_path = workflow_path_from_choice(subfolder, wf_choice)
            if wf_path is None:
                yield [], f"❌ Could not find workflow file for: {wf_choice}"
                return

            try:
                workflow = load_workflow(wf_path)
            except Exception as e:
                yield [], f"❌ Could not load workflow: {e}"
                return

            final_seed = int(sd) if int(sd) != -1 else random.randint(0, 2**31)
            wf  = inject_text2img_params(workflow, pos, neg, int(st), float(cf), int(w), int(h), final_seed)
            pid = queue_prompt(wf)

            if not pid:
                yield [], "❌ Failed to queue prompt — check ComfyUI logs."
                return

            yield [], f"⏳ Queued [{wf_choice}]  prompt_id={pid[:8]}…"

            images = wait_for_result(pid)
            if images:
                yield [str(p) for p in images], f"✅ Done!  {len(images)} image(s) — {wf_choice}"
            else:
                yield [], "⚠️ Generation finished but no images were returned."

        generate_btn.click(
            fn=run_generation,
            inputs=[workflow_dd, pos_prompt, neg_prompt, steps, cfg, width, height, seed],
            outputs=[output_gallery, status_box],
        )


def build_settings_tab():
    with gr.TabItem("⚙️ Settings"):
        gr.Markdown("### ComfyUI Connection")
        with gr.Row():
            host_inp = gr.Textbox(value=COMFYUI_HOST, label="ComfyUI Host")
            port_inp = gr.Number(value=COMFYUI_PORT,  label="Port", precision=0)
        check_btn   = gr.Button("Test Connection", variant="secondary")
        conn_status = gr.Textbox(label="Status", interactive=False)

        def do_check(host, port):
            global COMFYUI_HOST, COMFYUI_PORT, COMFYUI_URL, WS_URL
            COMFYUI_HOST = host
            COMFYUI_PORT = int(port)
            COMFYUI_URL  = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
            WS_URL       = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws"
            _, msg = check_connection()
            return msg

        check_btn.click(fn=do_check, inputs=[host_inp, port_inp], outputs=conn_status)

        gr.Markdown("---\n### Workflow folders")
        for label, subfolder in MODEL_TABS:
            gr.Textbox(
                value=str(WORKFLOWS_DIR / subfolder),
                label=f"{label} workflows",
                interactive=False,
            )
        gr.Markdown(
            "---\n"
            "**How to add a workflow:**\n"
            "1. In ComfyUI: gear icon → enable Dev Mode → **Save (API Format)**\n"
            "2. Drop the `.json` into the matching model folder above\n"
            "3. Click 🔄 Refresh on the tab (no restart needed)"
        )


# ───────────────────────────────────────────────
# CSS
# ───────────────────────────────────────────────
APP_CSS = """
footer { display: none !important; }

#aura-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px 4px 12px;
    border-bottom: 1px solid var(--border-color-primary);
    margin-bottom: 0 !important;
}
#aura-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0;
}
#aura-status {
    font-size: 0.85rem;
    opacity: 0.85;
    white-space: nowrap;
}

.gradio-container > .main > .wrap {
    padding-top: 0 !important;
}
"""


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def create_app() -> gr.Blocks:
    ok, status_msg = check_connection()

    with gr.Blocks(title="AuraCoreComfyUI") as demo:
        gr.HTML(f"""
            <div id="aura-header">
                <span id="aura-title">🎨 AuraCoreComfyUI</span>
                <span id="aura-status">{status_msg}</span>
            </div>
        """)

        with gr.Tabs():
            for label, subfolder in MODEL_TABS:
                build_model_tab(label, subfolder)
            build_settings_tab()

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )
