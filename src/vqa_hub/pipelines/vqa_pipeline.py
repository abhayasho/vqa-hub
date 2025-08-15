from __future__ import annotations
import argparse, yaml, json
from vqa_hub.router import Router
from vqa_hub.providers.huggingface import make_captioner, make_textgen
from vqa_hub.providers.ollama import make_ollama, HAS_OLLAMA
from vqa_hub.units.vision.blip_captioner import ImageCaptioner
from vqa_hub.units.vision.blip_vqa import BLIPVQA
from vqa_hub.units.vision.ocr_trocr import OCRTrOCR
from vqa_hub.units.text.gpt2_textgen import TextGenerator
from vqa_hub.units.fusion_simple import SimpleVQAFusion
from vqa_hub.hub import Hub
from transformers import pipeline as hf_pipeline
from vqa_hub.units.text.qa_small import ExtractiveQA
from vqa_hub.units.fusion_qa import FusionQA

def run_once(image: str, question: str, config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device")
    models = cfg.get("models", {})
    limits = cfg.get("limits", {})
    router_cfg = cfg.get("router", {})

    # Units
    cap_pipe = make_captioner(models.get("captioner", "Salesforce/blip-image-captioning-base"), device=device)
    captioner = ImageCaptioner(cap_pipe, models.get("captioner", "blip"))

    txt_pipe = make_textgen(models.get("textgen", "distilgpt2"), device=device)
    textgen = TextGenerator(txt_pipe, models.get("textgen", "distilgpt2"),
                            max_new_tokens=limits.get("textgen_max_new_tokens", 80))
    fusion = SimpleVQAFusion(text_model=textgen)

    # Extractive QA fusion (uses OCR+caption as context)
    qa_unit = ExtractiveQA(device=device)
    fusion_qa = FusionQA(qa_model=qa_unit)

    # Direct VQA
    vqa_model_name = models.get("vqa", "Salesforce/blip-vqa-base")
    try:
        vqa_pipe = hf_pipeline("vqa", model=vqa_model_name, device=device)
    except Exception:
        vqa_pipe = hf_pipeline("image-to-text", model=vqa_model_name, device=device)
    vqa_unit = BLIPVQA(vqa_pipe, vqa_model_name)

    # OCR
    ocr_unit = OCRTrOCR(model=models.get("ocr", "microsoft/trocr-base-printed"), device=device)

    # Router and optional big
    use_big = router_cfg.get("use_big_fallback", False)
    prefer_direct_vqa = router_cfg.get("prefer_direct_vqa", True)
    ocr_text_min_chars = int(router_cfg.get("ocr_text_min_chars", 12))
    big_model = None
    if use_big and HAS_OLLAMA:
        big_model = make_ollama(models.get("big_fallback", "llama3"))

    router = Router(use_big_fallback=use_big,
                    complex_cues=router_cfg.get("complex_cues", []),
                    prefer_direct_vqa=prefer_direct_vqa,
                    ocr_text_min_chars=ocr_text_min_chars)

    hub = Hub(captioner=captioner,
              fusion=fusion,
              router=router,
              vqa_unit=vqa_unit,
              ocr_unit=ocr_unit,
              big_model=big_model,
              ocr_text_min_chars=ocr_text_min_chars,
              fusion_qa=fusion_qa)

    result = hub.run_vqa(image_path=image, question=question)

    print("\n=== Final Answer ===")
    print(result.answer)

    short = {
        "router": result.trace.router.model_dump(),
        "caption": next((s.output.get("caption") for s in result.trace.steps if s.unit.startswith("captioner:")), ""),
    }
    print("\n=== Trace (short) ===")
    print(json.dumps(short, indent=2))

    return result

def cli_entry():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dump_trace", default=None)
    args = parser.parse_args()
    res = run_once(image=args.image, question=args.question, config_path=args.config)
    if args.dump_trace:
        with open(args.dump_trace, "w", encoding="utf-8") as f:
            f.write(res.trace.model_dump_json(indent=2))
        print(f"Full trace saved to {args.dump_trace}")

if __name__ == "__main__":
    cli_entry()
