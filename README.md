
# VQA Hub — fixed starter

A clean, working hub-and-unit multimodal framework for VQA with:
- Hub orchestrator
- Router with direct VQA vs OCR+fusion
- Units for BLIP captioning, BLIP VQA, TrOCR OCR, GPT-2 text fusion
- Config-driven setup
- Proper Python packaging (src layout) so `pip install -e .` works

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Replace the image path with a real photo on your machine
python -m vqa_hub.pipelines.vqa_pipeline   --image /path/to/your/photo.jpg   --question "What is happening?"   --config configs/default.yaml
```
If you prefer a short command, after editable install you can also use:
```bash
vqa --image /path/to/your/photo.jpg --question "What is happening?" --config configs/default.yaml
```
