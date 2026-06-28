"""
AuraCoreComfyUI - Gradio web frontend for ComfyUI

Connects to an already-running ComfyUI instance via its API.
Each .json file in the ./workflows/ folder becomes its own tab.

Usage:
    python app.py
    (opens at http://localhost:7860)

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

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# ComfyUI API helpers
# ─────────────────────────────────────────────
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
    """Check whether ComfyUI is reachable."""
    result = comfy_api("system_stats")
    if result:
        return True, f"✅ Connected to ComfyUI at {COMFYUI_URL}"
    return False, f"❌ Cannot reach ComfyUI at {COMFYUI_URL} — make sure it is running."


def queue_prompt(workflow: dict) -> str | None:
    """Submit a prompt to ComfyUI and return the prompt_id."""
    payload = {"prompt": workflow, "client_id": CLIENT_ID}
    result = comfy_api("prompt", payload)
    if result and "prompt_id" in result:
        return result["prompt_id"]
    log.error(f"Failed to queue prompt: {result}")
    return None


def wait_for_result(prompt_id: str, timeout: int = 300) -> list[Path]:
    """
    Listen via WebSocket until the prompt finishes.
    Returns list of local image Paths saved to OUTPUTS_DIR.
    """
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
                break  # generation complete
    finally:
        ws.close()

    # Fetch output images via history endpoint
    history = comfy_api(f"history/{prompt_id}")
    if not history or prompt_id not in history:
        return []

    outputs = history[prompt_id].get("outputs", {})
    saved = []
    for node_id, node_out in outputs.items():
        for img in node_out.get("images", []):
            fname     = img["filename"]
            subfolder = img.get("subfolder", "")
            folder    = img.get("type", "output")
            params    = urllib.parse.urlencode({"filename": fname, "subfolder": subfolder, "type": folder})
            img_url   = f"{COMFYUI_URL}/view?{params}"
            dest      = OUTPUTS_DIR / fname
            try:
                urllib.request.urlretrieve(img_url, dest)
                saved.append(dest)
            except Exception as e:
                log.error(f"Failed to save image {fname}: {e}")
    return saved


# ─────────────────────────────────────────────
# Workflow helpers
# ─────────────────────────────────────────────

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
    """
    Inject user parameters into the workflow dict.
    Finds nodes by class_type (heuristic approach).
    For complex workflows, target specific node IDs directly.
    """
    wf = json.loads(json.dumps(workflow))  # deep copy

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

    # First CLIPTextEncode = positive, second = negative
    clip_nodes = [(nid, n) for nid, n in wf.items()
                  if n.get("class_type") == "CLIPTextEncode"]
    if len(clip_nodes) >= 1:
        clip_nodes[0][1]["inputs"]["text"] = positive
    if len(clip_nodes) >= 2:
        clip_nodes[1][1]["inputs"]["text"] = negative

    return wf


def discover_workflow_files() -> list[Path]:
    """Return all .json files in the workflows/ directory, sorted."""
    WORKFLOWS_DIR.mkdir(exist_ok=True)
    return sorted(WORKFLOWS_DIR.glob("*.json"))


# ─────────────────────────────────────────────
# Tab builders
# ─────────────────────────────────────────────

def build_txt2img_tab(workflow_path: Path):
    """Generic text-to-image tab for a given workflow JSON."""
    wf_name = workflow_path.stem.replace("_", " ").replace("-", " ").title()

    with gr.TabItem(wf_name):
        gr.Markdown(f"### {wf_name}\nWorkflow file: `{workflow_path.name}`")

        with gr.Row():
            with gr.Column(scale=2):
                pos_prompt = gr.Textbox(
                    label="Positive Prompt",
                    placeholder="a cinematic photo of...",
                    lines=3,
                )
                neg_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, watermark...",
                    lines=2,
                )
                with gr.Row():
                    steps = gr.Slider(1, 60, value=20, step=1, label="Steps")
                    cfg   = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFG")
                with gr.Row():
                    width  = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                with gr.Row():
                    seed          = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                    rand_seed_btn = gr.Button("🎲 Randomize", size="sm")
                generate_btn = gr.Button("🎨 Generate", variant="primary")
                status_box   = gr.Textbox(label="Status", interactive=False, lines=1)

            with gr.Column(scale=3):
                output_gallery = gr.Gallery(
                    label="Output",
                    columns=2,
                    preview=True,
                    height=600,
                )

        def randomize_seed():
            return random.randint(0, 2**31)

        rand_seed_btn.click(fn=randomize_seed, outputs=seed)

        def run_generation(pos, neg, st, cf, w, h, sd):
            ok, msg = check_connection()
            if not ok:
                yield [], msg
                return

            try:
                workflow = load_workflow(workflow_path)
            except Exception as e:
                yield [], f"❌ Could not load workflow: {e}"
                return

            final_seed = int(sd) if int(sd) != -1 else random.randint(0, 2**31)
            wf  = inject_text2img_params(workflow, pos, neg, int(st), float(cf), int(w), int(h), final_seed)
            pid = queue_prompt(wf)

            if not pid:
                yield [], "❌ Failed to queue prompt — check ComfyUI logs."
                return

            yield [], f"⏳ Queued (prompt_id={pid[:8]}…). Waiting for result…"

            images = wait_for_result(pid)
            if images:
                yield [str(p) for p in images], f"✅ Done! {len(images)} image(s) generated."
            else:
                yield [], "⚠️ Generation finished but no images were returned."

        generate_btn.click(
            fn=run_generation,
            inputs=[pos_prompt, neg_prompt, steps, cfg, width, height, seed],
            outputs=[output_gallery, status_box],
        )


def build_settings_tab():
    """Settings tab for connection config."""
    with gr.TabItem("⚙️ Settings"):
        gr.Markdown("### ComfyUI Connection")
        with gr.Row():
            host_inp = gr.Textbox(value=COMFYUI_HOST, label="ComfyUI Host")
            port_inp = gr.Number(value=COMFYUI_PORT, label="Port", precision=0)
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

        gr.Markdown("---\n### Paths")
        gr.Textbox(value=str(WORKFLOWS_DIR), label="Workflows folder", interactive=False)
        gr.Textbox(value=str(OUTPUTS_DIR),   label="Outputs folder",   interactive=False)
        gr.Markdown(
            "💡 **Tip:** Drop workflow JSONs (exported via ComfyUI → Save API Format) "
            "into the `workflows/` folder and restart to get a new tab."
        )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def create_app() -> gr.Blocks:
    css = """
    footer { display: none !important; }
    .gr-button-primary { font-weight: 600; }
    """

    with gr.Blocks(title="AuraCoreComfyUI", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            "# 🎨 AuraCoreComfyUI\n"
            "Gradio frontend for your local ComfyUI instance."
        )

        ok, status_msg = check_connection()
        gr.Markdown(f"**ComfyUI:** {status_msg}")

        workflows = discover_workflow_files()
        log.info(f"Found {len(workflows)} workflow(s): {[w.name for w in workflows]}")

        with gr.Tabs():
            if workflows:
                for wf_path in workflows:
                    build_txt2img_tab(wf_path)
            else:
                with gr.TabItem("📂 No Workflows Found"):
                    gr.Markdown(
                        "**No workflow JSON files found.**\n\n"
                        f"Place your ComfyUI workflow files (saved as API Format) inside:\n\n"
                        f"`{WORKFLOWS_DIR}`\n\n"
                        "In ComfyUI: gear icon → enable Dev Mode → **Save (API Format)**."
                    )
            build_settings_tab()

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )