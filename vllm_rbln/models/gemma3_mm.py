from collections.abc import Mapping
from typing import Optional, Union

import torch
from vllm.model_executor.models.gemma3_mm import Gemma3MultiModalProcessor
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalInputs


class RBLNGemma3MultiModalProcessor(Gemma3MultiModalProcessor):
    
    def _pad_for_gemma3(self, prompt_ids: list[int], prompt: str):
        token_type_ids = torch.tensor(prompt_ids)==self.info.get_hf_processor().image_token_id
        
        image_prefill_chunk_size = self.info.get_hf_processor().image_seq_length
        # Find image start positions
        image_starts = [
            s
            for s in torch.where(token_type_ids)[0]
            if torch.all(token_type_ids[s : s + image_prefill_chunk_size])
        ]
        padded_seq_len = 0  
        for image_start in image_starts:
            pad_needed = image_prefill_chunk_size - (image_start + padded_seq_len) % image_prefill_chunk_size
            padded_seq_len += pad_needed
        
        pad_token = "<unused6241>"
        pad_token_id = 262143
        
        prompt_ids = prompt_ids + [pad_token_id] * padded_seq_len
        prompt = prompt + pad_token * padded_seq_len
        return prompt_ids, prompt
    
    
    def apply(self, *args, **kwargs):
        output = super().apply(*args, **kwargs)
        prompt_ids, prompt = self._pad_for_gemma3(output["prompt_token_ids"], output["prompt"])
        
        output["prompt_token_ids"] = prompt_ids
        output["prompt"] = prompt
        
        return output