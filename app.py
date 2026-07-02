"""
AuraCoreComfyUI - Gradio web frontend for ComfyUI

Each tab = one AI model.
Inside each tab, a dropdown selects which workflow JSON to run.
Workflows live in: ./workflows/<model_folder>/<workflow>.json

Sidecar config system:
  Each workflow can have an optional <workflow_stem>_config.json next to it.
  This JSON tells the injector exactly which node IDs / fields to patch,
  handling exotic workflows (custom prompt nodes, no negative prompt,
  aspect-ratio resolution selectors, etc.) without touching app.py.

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
from typing import Any, Generator

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
# ───────────────────────────────────────────────
MODEL_TABS = [
    ("Krea 2",       "krea2"),
    ("Qwen Image",   "qwen_image"),
    ("Ideogram 4",   "ideogram4"),
    ("Klein 9B",     "klein9b"),
]

ASPECT_RATIO_OPTIONS = [
    "1:1 (Square)",
    "2:3 (Portrait)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]


# ───────────────────────────────────────────────
# ComfyUI API helpers
# ───────────────────────────────────────────────
CLIENT_ID = str(uuid.uuid4())


def comfy_api(endpoint: str, data: dict | None = None) -> Any:
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


def _build_progress_bar(pct: int) -> str:
    """Return an HTML progress bar string for the status strip."""
    filled = round(pct / 5)   # 20 blocks total (0–100 → 0–20)
    empty  = 20 - filled
    bar    = "█" * filled + "░" * empty
    return (
        f'<p class="aura-status">'  
        f'<span class="aura-bar">{bar}</span>'  
        f'&nbsp; Generating… {pct}%'  
        f'</p>'
    )


def stream_generation(
    prompt_id: str,
    timeout: int = 300,
) -> Generator[str | None, None, list[Path]]:
    """
    Generator that:
      - yields HTML status strings as ComfyUI sends progress frames
      - returns the list of saved image Paths when done

    Usage (inside an outer generator):
        gen = stream_generation(pid)
        images = None
        try:
            while True:
                status_html = next(gen)
                yield status_html          # push to Gradio
        except StopIteration as e:
            images = e.value
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
            except Exception:
                break

            if isinstance(raw, bytes):
                # Binary preview frames — ignore
                continue

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            mtype = msg.get("type", "")
            data  = msg.get("data", {})

            if mtype == "progress":
                value = data.get("value", 0)
                max_v = data.get("max", 1) or 1
                pct   = round(value / max_v * 100)
                yield _build_progress_bar(pct)

            elif mtype == "executing":
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    # Generation complete
                    break
    finally:
        ws.close()

    # Fetch output images
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
    return sorted(f for f in folder.glob("*.json") if not f.stem.endswith("_config"))


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


def load_sidecar_config(workflow_path: Path) -> dict | None:
    config_path = workflow_path.parent / (workflow_path.stem + "_config.json")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            log.info(f"Loaded sidecar config: {config_path.name}")
            return cfg
        except Exception as e:
            log.warning(f"Could not load sidecar config {config_path.name}: {e}")
    return None


def has_aspect_ratio_config(workflow_path: Path) -> bool:
    cfg = load_sidecar_config(workflow_path)
    if cfg and "resolution_node" in cfg:
        return cfg["resolution_node"].get("mode") == "aspect_ratio"
    return False


def has_negative_prompt(workflow_path: Path) -> bool:
    cfg = load_sidecar_config(workflow_path)
    if cfg is not None:
        return cfg.get("negative_prompt", True) is not False
    return True


def get_resolution_defaults(workflow_path: Path | None) -> tuple[str, float]:
    """Read the default aspect ratio / megapixels for a workflow's resolution_node."""
    if workflow_path:
        cfg = load_sidecar_config(workflow_path)
        if cfg and "resolution_node" in cfg:
            rn = cfg["resolution_node"]
            return (
                rn.get("default_aspect_ratio", "1:1 (Square)"),
                rn.get("default_megapixels", 1.0),
            )
    return "1:1 (Square)", 1.0


def get_resolution_multiple(workflow_path: Path | None) -> int:
    if workflow_path:
        cfg = load_sidecar_config(workflow_path)
        if cfg and "resolution_node" in cfg:
            return cfg["resolution_node"].get("multiple", 8)
    return 8


def compute_true_resolution(aspect_ratio: str, megapixels: float, multiple: int = 8) -> tuple[int, int]:
    """
    Approximate the width/height that a ComfyUI ResolutionSelector-style node
    would produce for a given aspect ratio label and megapixel budget.

    NOTE: this mirrors the common "solve for scale that hits the target
    megapixel count, then round each side to the nearest multiple" approach
    used by most ComfyUI resolution-selector custom nodes. If the actual
    ResolutionSelector node in your ComfyUI install rounds differently, the
    real output may be off by a `multiple` or two from what's shown here.
    """
    try:
        ratio_part = aspect_ratio.split("(")[0].strip()  # "16:9"
        rw_str, rh_str = ratio_part.split(":")
        rw, rh = float(rw_str), float(rh_str)
    except Exception:
        rw, rh = 1.0, 1.0

    total_pixels = max(megapixels, 0.01) * 1_000_000
    scale = (total_pixels / (rw * rh)) ** 0.5

    width  = max(multiple, round((rw * scale) / multiple) * multiple)
    height = max(multiple, round((rh * scale) / multiple) * multiple)
    return int(width), int(height)


# ───────────────────────────────────────────────
# Injection engine
# ───────────────────────────────────────────────

def inject_params(
    workflow: dict,
    workflow_path: Path,
    positive: str,
    negative: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    aspect_ratio: str = "1:1 (Square)",
    megapixels: float = 1.0,
) -> dict:
    wf       = json.loads(json.dumps(workflow))
    cfg_data = load_sidecar_config(workflow_path)

    if cfg_data:
        pn = cfg_data.get("prompt_node")
        if pn:
            wf[pn["id"]]["inputs"][pn["field"]] = positive

        neg_cfg = cfg_data.get("negative_prompt", True)
        if neg_cfg and isinstance(neg_cfg, dict):
            wf[neg_cfg["id"]]["inputs"][neg_cfg["field"]] = negative

        sn = cfg_data.get("seed_node")
        if sn:
            wf[sn["id"]]["inputs"][sn["field"]] = seed if seed != -1 else random.randint(0, 2**31)

        st = cfg_data.get("steps_node")
        if st:
            wf[st["id"]]["inputs"][st["field"]] = steps

        cf = cfg_data.get("cfg_node")
        if cf:
            wf[cf["id"]]["inputs"][cf["field"]] = cfg

        rn = cfg_data.get("resolution_node")
        if rn and rn.get("mode") == "aspect_ratio":
            wf[rn["id"]]["inputs"][rn["aspect_ratio_field"]] = aspect_ratio
            wf[rn["id"]]["inputs"][rn["megapixels_field"]]   = megapixels
        elif rn and rn.get("mode", "pixels") == "pixels":
            wf[rn["id"]]["inputs"][rn.get("width_field",  "width")]  = width
            wf[rn["id"]]["inputs"][rn.get("height_field", "height")] = height
    else:
        for node_id, node in wf.items():
            ctype = node.get("class_type", "")
            if ctype in ("KSampler", "KSamplerAdvanced"):
                inp = node["inputs"]
                inp["steps"]   = steps
                inp["cfg"]     = cfg
                inp["seed"]    = seed if seed != -1 else random.randint(0, 2**31)
                inp["denoise"] = inp.get("denoise", 1.0)
            if ctype in ("EmptyLatentImage", "EmptySD3LatentImage"):
                inp = node["inputs"]
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

    first_path    = workflow_path_from_choice(subfolder, choices[0]) if choices[0] != "(no workflows yet)" else None
    init_ar_mode  = has_aspect_ratio_config(first_path) if first_path else False
    init_show_neg = has_negative_prompt(first_path)     if first_path else True
    init_ar, init_mp = get_resolution_defaults(first_path)
    init_multiple    = get_resolution_multiple(first_path)
    init_w, init_h   = compute_true_resolution(init_ar, init_mp, init_multiple)

    with gr.TabItem(label):

        # Row 1: workflow selector + buttons
        with gr.Row(equal_height=True):
            workflow_dd = gr.Dropdown(
                choices=choices,
                value=choices[0],
                label="",
                show_label=False,
                scale=4,
                interactive=True,
            )
            with gr.Column(scale=1, min_width=110):
                reload_btn   = gr.Button("🔄 Refresh",  size="sm")
                generate_btn = gr.Button("🎨 Generate", variant="primary", size="sm")

        # Row 2: left controls | right (status + gallery)
        with gr.Row():

            # ── Left: prompt + params ──────────────────────────
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
                    visible=init_show_neg,
                )
                # Hidden width/height sliders kept only so run_generation's
                # signature stays workflow-agnostic; not shown for aspect-ratio
                # workflows like Krea 2.
                with gr.Row(visible=not init_ar_mode) as pixel_row:
                    width  = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                with gr.Row(visible=init_ar_mode) as ar_row:
                    megapixels_sl = gr.Number(
                        value=init_mp,
                        minimum=0.1,
                        maximum=8.0,
                        step=0.05,
                        label="Megapixels",
                        scale=2,
                    )
                    aspect_ratio_dd = gr.Dropdown(
                        choices=ASPECT_RATIO_OPTIONS,
                        value=init_ar,
                        label="Aspect Ratio",
                        scale=3,
                    )
                with gr.Row(visible=init_ar_mode) as res_info_row:
                    resolution_info = gr.Textbox(
                        value=f"Output resolution: {init_w} × {init_h} px",
                        label="True Resolution",
                        interactive=False,
                    )
                with gr.Row():
                    steps = gr.Slider(1, 60,    value=8,   step=1,   label="Steps")
                    cfg   = gr.Slider(1.0, 20.0, value=1.0, step=0.5, label="CFG")
                with gr.Row():
                    seed          = gr.Number(value=-1, label="Seed  (−1 = random)", precision=0)
                    rand_seed_btn = gr.Button("🎲 Randomize", size="sm")

            # ── Right: status HTML strip, then gallery ─────────────
            with gr.Column(scale=3):
                status_html = gr.HTML(
                    value='<p class="aura-status">Ready.</p>',
                )
                output_gallery = gr.Gallery(
                    label="Output",
                    columns=1,
                    rows=1,
                    preview=True,
                    selected_index=0,
                    height=620,
                    object_fit="contain",
                )

        # ── Event handlers ─────────────────────────────────

        def on_workflow_change(wf_choice):
            wf_path  = workflow_path_from_choice(subfolder, wf_choice)
            ar_mode  = has_aspect_ratio_config(wf_path) if wf_path else False
            show_neg = has_negative_prompt(wf_path)     if wf_path else True
            ar, mp   = get_resolution_defaults(wf_path)
            mult     = get_resolution_multiple(wf_path)
            w, h     = compute_true_resolution(ar, mp, mult)
            return (
                gr.Row(visible=not ar_mode),
                gr.Row(visible=ar_mode),
                gr.Row(visible=ar_mode),
                gr.Textbox(visible=show_neg),
                gr.Dropdown(value=ar),
                gr.Number(value=mp),
                gr.Textbox(value=f"Output resolution: {w} × {h} px"),
            )

        workflow_dd.change(
            fn=on_workflow_change,
            inputs=workflow_dd,
            outputs=[
                pixel_row, ar_row, res_info_row, neg_prompt,
                aspect_ratio_dd, megapixels_sl, resolution_info,
            ],
        )

        def on_resolution_inputs_change(ar, mp, wf_choice):
            wf_path = workflow_path_from_choice(subfolder, wf_choice)
            mult    = get_resolution_multiple(wf_path)
            w, h    = compute_true_resolution(ar, mp or 0, mult)
            return f"Output resolution: {w} × {h} px"

        aspect_ratio_dd.change(
            fn=on_resolution_inputs_change,
            inputs=[aspect_ratio_dd, megapixels_sl, workflow_dd],
            outputs=resolution_info,
        )
        megapixels_sl.change(
            fn=on_resolution_inputs_change,
            inputs=[aspect_ratio_dd, megapixels_sl, workflow_dd],
            outputs=resolution_info,
        )

        reload_btn.click(
            fn=lambda: gr.Dropdown(
                choices=workflow_choices(subfolder),
                value=workflow_choices(subfolder)[0],
            ),
            outputs=workflow_dd,
        )

        rand_seed_btn.click(fn=lambda: random.randint(0, 2**31), outputs=seed)

        def _status(msg: str) -> str:
            return f'<p class="aura-status">{msg}</p>'

        def run_generation(wf_choice, pos, neg, st, cf, w, h, sd, ar, mp):
            ok, msg = check_connection()
            if not ok:
                yield gr.Gallery(value=[], selected_index=None), _status(msg)
                return

            if wf_choice == "(no workflows yet)":
                yield gr.Gallery(value=[], selected_index=None), _status(f"❌ No workflow — add a JSON to workflows/{subfolder}/")
                return

            wf_path = workflow_path_from_choice(subfolder, wf_choice)
            if wf_path is None:
                yield gr.Gallery(value=[], selected_index=None), _status(f"❌ Cannot find workflow: {wf_choice}")
                return

            try:
                workflow = load_workflow(wf_path)
            except Exception as e:
                yield gr.Gallery(value=[], selected_index=None), _status(f"❌ Cannot load workflow: {e}")
                return

            final_seed = int(sd) if int(sd) != -1 else random.randint(0, 2**31)
            wf = inject_params(
                workflow, wf_path,
                positive=pos, negative=neg,
                steps=int(st), cfg=float(cf),
                width=int(w), height=int(h),
                seed=final_seed,
                aspect_ratio=ar, megapixels=float(mp),
            )

            pid = queue_prompt(wf)
            if not pid:
                yield gr.Gallery(value=[], selected_index=None), _status("❌ Failed to queue — check ComfyUI logs.")
                return

            # Show initial 0% bar immediately
            yield gr.Gallery(value=[], selected_index=None), _build_progress_bar(0)

            # Stream progress updates from ComfyUI WebSocket
            gen = stream_generation(pid)
            images = []
            try:
                while True:
                    html = next(gen)
                    yield gr.Gallery(value=[], selected_index=None), html
            except StopIteration as e:
                images = e.value  # list[Path] returned by the generator

            if images:
                paths = [str(p) for p in images]
                yield gr.Gallery(value=paths, selected_index=0), _status(f"✅ Done — {len(images)} image(s)")
            else:
                yield gr.Gallery(value=[], selected_index=None), _status("⚠️ Done but no images returned.")

        generate_btn.click(
            fn=run_generation,
            inputs=[
                workflow_dd, pos_prompt, neg_prompt,
                steps, cfg, width, height, seed,
                aspect_ratio_dd, megapixels_sl,
            ],
            outputs=[output_gallery, status_html],
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
            "3. If needed, create a `<workflow_name>_config.json` sidecar next to it\n"
            "4. Click 🔄 Refresh on the tab (no restart needed)\n\n"
            "**Sidecar config keys:** `prompt_node`, `negative_prompt` (false to hide), "
            "`seed_node`, `steps_node`, `cfg_node`, `resolution_node` "
            "(`default_aspect_ratio`, `default_megapixels`, `multiple` for the rounding "
            "step used by the True Resolution readout)"
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
#aura-title  { font-size: 1.3rem; font-weight: 700; margin: 0; }
#aura-status { font-size: 0.85rem; opacity: 0.85; white-space: nowrap; }

.gradio-container > .main > .wrap { padding-top: 0 !important; }

/* Status strip directly above the gallery */
.aura-status {
    margin: 0 0 4px 0 !important;
    padding: 3px 10px !important;
    font-size: 0.82rem !important;
    opacity: 0.88;
    border-radius: 4px;
    background: var(--background-fill-secondary, rgba(0,0,0,0.04));
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: monospace;
}

/* Block characters in the progress bar render crisply */
.aura-bar {
    letter-spacing: 1px;
    color: var(--color-accent, #4f98a3);
}

/* Gallery */
.gradio-gallery .preview-container,
.gradio-gallery .preview-container img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}
"""


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def create_app() -> gr.Blocks:
    ok, status_msg = check_connection()

    with gr.Blocks(title="AuraCoreComfyUI", css=APP_CSS) as demo:
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
    )
