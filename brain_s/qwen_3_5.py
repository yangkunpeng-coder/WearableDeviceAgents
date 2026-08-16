import torch
from typing import Literal
from transformers import AutoProcessor, AutoModelForImageTextToText
from base_class.brain_base import BrainBase


class Qwen3_5Brain(BrainBase):
    def __init__(self, model_name: Literal['qwen3.5_0.8B'], **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name == 'qwen3.5_0.8B':
            model_path = r'D:\work\python_project\WearableDeviceAgents\model\qwen3.5-0.8B'
            self.brain = AutoModelForImageTextToText.from_pretrained(model_path).to(device)
            self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self, sys_prompt, user_prompt, formating=None, **kwargs):
        output_dict = {}
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sys_prompt + '\n' + user_prompt}
                ]
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.brain.device)
        outputs = self.brain.generate(**inputs, max_new_tokens=10000, pad_token_id=self.processor.tokenizer.pad_token_id)
        output = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]).replace('<|im_end|>', '').replace('<|endoftext|>', '')
        print(output)
        if output is not None:
            if formating is not None:
                output_dict = formating.model_validate_json(output)
            else:
                output_dict = output
        return output_dict
