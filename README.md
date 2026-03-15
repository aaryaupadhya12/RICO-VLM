# Qwen2.5-VL Fine-tuned on RICO Screen2Words

# ScreeScribeVLM — Mobile UI Screen Captioning with Qwen2.5-VL

![Fine-tuning](https://img.shields.io/badge/Method-QLoRA-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--7B-purple)
![Dataset](https://img.shields.io/badge/Dataset-RICO--Screen2Words-orange)
![Hardware](https://img.shields.io/badge/Hardware-T4%20GPU-yellow)
![Library](https://img.shields.io/badge/Library-Unsloth-red)
![Samples](https://img.shields.io/badge/Training%20Samples-700-green)
![Epochs](https://img.shields.io/badge/Epochs-1-blue)
![Steps](https://img.shields.io/badge/Steps-88-blueviolet)
![Loss](https://img.shields.io/badge/Train%20Loss-0.30-brightgreen)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-aaryaupadhya20-ff69b4)

Fine-tuned `Qwen2.5-VL-7B-Instruct` on the RICO Screen2Words dataset to generate
natural language descriptions of mobile app UI screens.

> For full training decisions, dataset rationale, loss analysis, and inference results see [`Description.md`](Description.md).

## Repository Structure

```
MultiModal_Qwen2.7B_Finetuning/
├── Config/
│   └── requirementx.txt
├── src/
│   ├── Full_epoch_runs/
│   │   └── qwen_2_5_vl_7b_finetuning_Full_trai...
│   ├── Inference_trial/
│   │   ├── image.png
│   │   └── inference.py
│   └── max_Step_check/
│       └── qwen_2_5_vl_7b_finetuning.py
├── .gitignore
├── Description.md
├── LICENSE
└── README.md
```

## Model Details
- **Base model:** unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit
- **Dataset:** rootsautomation/RICO-Screen2Words
- **Method:** QLoRA (4-bit) via Unsloth
- **Task:** Vision → Text (UI screen captioning)

## Training Hyperparameters
| Parameter | Value |
|---|---|
| Learning rate | 2e-4 |
| Epochs | 1 |
| Batch size | 2 |
| Gradient accumulation | 4 (effective batch = 8) |
| Scheduler | cosine |
| Optimizer | adamw_8bit |
| LoRA rank | 16 |
| Seed | 3407 |

## How to Run Inference

Install dependencies:

```bash
pip install Config/requirementx.txt
```

Login to Hugging Face (required to pull the model):

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")  # get token from https://huggingface.co/settings/tokens
```

Run inference:

```python
from unsloth import FastVisionModel
import torch
from PIL import Image

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "aaryaupadhya20/rico-screen2words-qwen2.5_7B_VL",
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = torch.float16,
)
FastVisionModel.for_inference(model)

image = Image.open("your_screen.png")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are a UI/UX expert. Describe what you see on this mobile app screen."},
            {"type": "image", "image": image},
        ]
    }
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

output_ids = model.generate(**inputs, max_new_tokens=128, use_cache=False)

prediction = tokenizer.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print(prediction)
```

A ready-to-run script is available at [`src/Inference_trial/inference.py`](src/Inference_trial/inference.py).

Future: Run 3-5 Epochs with 50% of the dataset for better results on better compute availability
