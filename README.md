# Qwen2.5-VL Fine-tuned on RICO Screen2Words

# ScreenScribe — Mobile UI Screen Captioning with Qwen2.5-VL

![Fine-tuning](https://img.shields.io/badge/Method-QLoRA-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--7B-purple)
![Dataset](https://img.shields.io/badge/Dataset-RICO--Screen2Words-orange)
![Hardware](https://img.shields.io/badge/Hardware-T4%20GPU-yellow)
![Library](https://img.shields.io/badge/Library-Unsloth-red)
![Samples](https://img.shields.io/badge/Training%20Samples-700-green)
![Epochs](https://img.shields.io/badge/Epochs-3-blue)
![Steps](https://img.shields.io/badge/Steps-264-blueviolet)
![Loss](https://img.shields.io/badge/Train%20Loss-0.72-brightgreen)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-aaryaupadhya20-ff69b4)

Fine-tuned `Qwen2.5-VL-7B-Instruct` on the RICO Screen2Words dataset to generate 
natural language descriptions of mobile app UI screens.

## Model Details
- **Base model:** unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit
- **Dataset:** rootsautomation/RICO-Screen2Words
- **Method:** QLoRA (4-bit) via Unsloth
- **Task:** Vision → Text (UI screen captioning)

## Training Hyperparameters
| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Epochs | 3 |
| Batch size | 2 |
| Gradient accumulation | 4 (effective batch = 8) |
| Scheduler | cosine |
| Optimizer | adamw_8bit |
| LoRA rank | 16 |
| Seed | 42 |

## How to Run Inference
```python
from unsloth import FastVisionModel
from PIL import Image

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "aaryaupadhya20/rico-screen2words-qwen25vl",
    load_in_4bit = True,
)
FastVisionModel.for_inference(model)

image = Image.open("your_screen.png")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "You are a UI/UX expert. Describe what you see on this mobile app screen."},
        ]
    }
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)
```


