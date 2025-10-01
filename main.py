from transformers import pipeline
from PIL import Image
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

# Local image path (no need to use requests here)
image_path = "your image path"
image = Image.open(image_path)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray"},
            {"type": "image", "image": image},
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
