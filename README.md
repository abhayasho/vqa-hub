# VQA Hub

A modular **hub-and-unit model framework** for **Visual Question Answering (VQA)** research.  
This project explores whether multiple smaller, specialized text and vision models can work together in a back-and-forth process to rival large multimodal models.

---

## 📌 Overview
The **Hub** coordinates multiple **unit models**:
- **Text models** (e.g., GPT-2) for reasoning and question understanding
- **Vision models** (e.g., Ollama BLIP captioner) for image description
- **Fusion models** for combining text + vision data

The hub routes tasks between unit models, passing intermediate results until an answer is ready.

---

## 🧠 Goal
Evaluate whether a distributed chain of small models can:
- Match or exceed the accuracy of large multimodal models
- Reduce computation costs
- Provide more flexible, interpretable reasoning steps

---

## 🏗 Architecture
```
Image → Vision Model → Hub → Text Model → Hub → (repeat until answer) → Output
```

**Technologies Used**:
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)
- Python 3.10+

---

## 🚀 Installation
```bash
git clone https://github.com/abhayasho/vqa-hub.git
cd vqa-hub
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 📷 Example Usage
```bash
python -m vqa_hub.pipelines.vqa_pipeline   --image ./demo.jpg   --question "What is happening?"   --config configs/default.yaml
```

---

## 📂 Project Structure
```
vqa-hub/
│── configs/           # YAML configs for models and pipelines
│── data/              # Sample images
│── src/vqa_hub/       # Source code
│── requirements.txt   # Dependencies
│── README.md          # Project documentation
```

---

## 🧪 Research Context
This framework is part of a research project investigating **hub-and-unit model orchestration** for multimodal tasks, focusing on **VQA** as a benchmark.

---

## 📜 License
MIT License — feel free to use and modify for research or development.
