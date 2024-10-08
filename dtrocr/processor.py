from transformers import GPT2Tokenizer, AutoImageProcessor

# AraBERT preprocessor
from arabert.preprocess import ArabertPreprocessor

from PIL import Image
from typing import List, Union
from dtrocr.config import DTrOCRConfig
from dtrocr.data import DTrOCRProcessorOutput


class DTrOCRProcessor:
    def __init__(self, config: DTrOCRConfig, add_bos_token: bool = False, add_eos_token: bool = False):
        self.vit_processor = AutoImageProcessor.from_pretrained(
            config.vit_hf_model,
            size={
                "height": config.image_size[0],
                'width': config.image_size[1]
            },
            use_fast=True
        )
        self.preprocessor = ArabertPreprocessor(
            model_name=config.gpt2_hf_model
        ) if config.lang=='ar' else None
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            config.gpt2_hf_model,
            add_bos_token=add_bos_token,
            model_max_length=config.max_position_embeddings - int(
                (config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1])
            )
        )
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.add_eos_token = add_eos_token

        # Bind a new method to gpt2_tokenizer
        self.tokenizer.build_inputs_with_special_tokens = modified_build_inputs_with_special_tokens.__get__(
            self.tokenizer
        )
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        texts: Union[str, List[str]] = None,
        return_labels: bool = False,
        input_data_format: str = 'channels_last',
        padding: Union[bool, str] = False,
        *args,
        **kwargs
    ) -> DTrOCRProcessorOutput:
        # Clean arabic text
        
        # check for bos token
        if self.preprocessor and texts is not None:
            if texts != self.tokenizer.bos_token:
                # batch handeling
                if isinstance(texts, list):
                    texts = [
                        self.preprocessor.preprocess(t) for t in texts
                    ]
                elif isinstance(texts, str):
                    texts = self.preprocessor.preprocess(texts)
                    
        text_inputs = self.tokenizer(
            texts, padding=padding, *args, **kwargs
        ) if texts is not None else None

        image_inputs = self.vit_processor(
            images, input_data_format=input_data_format, *args, **kwargs
        ) if images is not None else None

        return DTrOCRProcessorOutput(
            pixel_values=image_inputs["pixel_values"] if images is not None else None,
            input_ids=text_inputs['input_ids'] if texts is not None else None,
            attention_mask=text_inputs['attention_mask'] if texts is not None else None,
            labels=text_inputs['input_ids'] if texts is not None and return_labels else None
        )


def modified_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if self.add_bos_token:
        bos_token_ids = [self.bos_token_id]
    else:
        bos_token_ids = []

    if self.add_eos_token:
        eos_token_ids = [self.eos_token_id]
    else:
        eos_token_ids = []

    output = bos_token_ids + token_ids_0 + eos_token_ids

    if token_ids_1 is None:
        return output

    return output + bos_token_ids + token_ids_1
