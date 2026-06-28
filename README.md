# AuraCoreComfyUI

A clean Gradio web frontend for your local ComfyUI installation.
Each workflow JSON = its own tab. No node graph zooming required.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Opens at http://localhost:7860
```

## How to Add Workflows

1. In ComfyUI, click gear icon → enable **Dev Mode**
2. Click **Save (API Format)**
3. Drop the `.json` into the `workflows/` folder
4. Restart `app.py` — new tab appears automatically

## Project Structure
