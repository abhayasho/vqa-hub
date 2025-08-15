from __future__ import annotations
from typing import Optional

from .schemas import StepTrace, RouterInfo, RunTrace, HubResult
from .router import Router
from .utils.io import load_image

from .units.vision.blip_captioner import ImageCaptioner
from .units.vision.blip_vqa import BLIPVQA
from .units.vision.ocr_trocr import OCRTrOCR
from .units.fusion_simple import SimpleVQAFusion
from .units.fusion_qa import FusionQA


class Hub:
    def __init__(
        self,
        captioner: ImageCaptioner,
        fusion: SimpleVQAFusion,
        router: Router,
        vqa_unit: Optional[BLIPVQA] = None,
        ocr_unit: Optional[OCRTrOCR] = None,
        big_model: Optional[object] = None,
        ocr_text_min_chars: int = 12,
        fusion_qa: Optional[FusionQA] = None,
    ):
        self.captioner = captioner
        self.fusion = fusion
        self.router = router
        self.vqa_unit = vqa_unit
        self.ocr_unit = ocr_unit
        self.big_model = big_model
        self.ocr_text_min_chars = int(ocr_text_min_chars)
        self.fusion_qa = fusion_qa

    def run_vqa(self, image_path: str, question: str) -> HubResult:
        img = load_image(image_path)
        decision = self.router.decide(question)
        trace = RunTrace(
            image=image_path,
            question=question,
            router=RouterInfo(decision=decision.path, reason=decision.reason),
            steps=[],
        )

        # Step 1: caption
        cap_out = self.captioner.run(img)
        caption = cap_out["caption"]
        trace.steps.append(StepTrace(unit=self.captioner.name, output=cap_out))

        # Branch based on router decision
        if "ocr" in decision.path and self.ocr_unit is not None:
            ocr_out = self.ocr_unit.run(img)
            ocr_text = ocr_out.get("text", "")
            trace.steps.append(StepTrace(unit=self.ocr_unit.name, output=ocr_out))

            if self.fusion_qa is not None:
                # Use extractive QA fusion grounded on caption + OCR text
                fusion_out = self.fusion_qa.run(caption=caption, question=question, ocr_text=ocr_text)
                answer = fusion_out["answer_text"].strip()
                trace.steps.append(StepTrace(unit=self.fusion_qa.name, output=fusion_out))
            else:
                # Fallback to simple text generation fusion
                context = (
                    f"Caption: {caption}. On-image text: {ocr_text}."
                    if len(ocr_text) >= self.ocr_text_min_chars
                    else f"Caption: {caption}."
                )
                fusion_out = self.fusion.run(caption=context, question=question)
                answer = fusion_out["answer_text"].strip()
                trace.steps.append(StepTrace(unit=self.fusion.name, output=fusion_out))

        elif "vqa" in decision.path and self.vqa_unit is not None:
            vqa_out = self.vqa_unit.run(image=img, question=question)
            answer = vqa_out.get("answer", "").strip()
            trace.steps.append(StepTrace(unit=self.vqa_unit.name, output=vqa_out))

        else:
            # Default fusion using caption only
            fusion_out = self.fusion.run(caption=caption, question=question)
            answer = fusion_out["answer_text"].strip()
            trace.steps.append(StepTrace(unit=self.fusion.name, output=fusion_out))

        # Optional big fallback
        if self.big_model and "maybe_big" in decision.path:
            if self._low_confidence(answer):
                prompt = self._build_big_prompt(caption, question, answer)
                big_answer = self.big_model.invoke(prompt).strip()  # langchain llm style
                trace.steps.append(StepTrace(unit="ollama", output={"text": big_answer, "prompt": prompt}))
                if self._better_than_small(big_answer, answer):
                    answer = big_answer

        return HubResult(answer=answer, trace=trace)

    @staticmethod
    def _low_confidence(ans: str) -> bool:
        uncertain = ["not sure", "cannot tell", "unclear", "unknown", "no idea"]
        if len(ans.split()) < 2:
            return True
        return any(u in ans.lower() for u in uncertain)

    @staticmethod
    def _better_than_small(big: str, small: str) -> bool:
        return len(big) > len(small) and "not sure" not in big.lower()

    @staticmethod
    def _build_big_prompt(caption: str, question: str, small_answer: str) -> str:
        return (
            "You are a helpful visual question answering assistant.\n"
            f"Caption: {caption}\n"
            f"Question: {question}\n"
            f"Small model answer: {small_answer}\n"
            "Give a better answer if possible, or say the small answer is correct."
        )
