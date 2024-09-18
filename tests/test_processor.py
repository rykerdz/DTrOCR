from dtrocr.config import DTrOCRConfig
from dtrocr.processor import DTrOCRProcessor

import random
from PIL import Image


def test_tokeniser_with_bos_token():
    tokenizer = DTrOCRProcessor(config=DTrOCRConfig(lang='en'), add_bos_token=True)
    tokeniser_output = tokenizer(texts=["This is a sentence", "That is not a sentence, sorry"])

    expected_input_ids = [
        [50256, 1212, 318, 257, 6827],
        [50256, 2504, 318, 407, 257, 6827, 11, 7926]
    ]
    expected_attention_mask = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
    
    assert tokeniser_output.input_ids == expected_input_ids
    assert tokeniser_output.attention_mask == expected_attention_mask

def test_tokeniser_with_eos_token():
    tokenizer = DTrOCRProcessor(config=DTrOCRConfig(lang='en'), add_eos_token=True)
    tokeniser_output = tokenizer(texts=["This is a sentence", "That is not a sentence, sorry"])

    expected_input_ids = [
        [1212, 318, 257, 6827, 50256],
        [2504, 318, 407, 257, 6827, 11, 7926, 50256]
    ]
    expected_attention_mask = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]

    assert tokeniser_output.input_ids == expected_input_ids
    assert tokeniser_output.attention_mask == expected_attention_mask


def test_tokeniser_with_eos_and_bos_tokens():
    tokenizer = DTrOCRProcessor(config=DTrOCRConfig(lang='en'), add_bos_token=True, add_eos_token=True)
    tokeniser_output = tokenizer(texts=["This is a sentence", "That is not a sentence, sorry"])

    expected_input_ids = [
        [50256, 1212, 318, 257, 6827, 50256],
        [50256, 2504, 318, 407, 257, 6827, 11, 7926, 50256]
    ]
    expected_attention_mask = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

    assert tokeniser_output.input_ids == expected_input_ids
    assert tokeniser_output.attention_mask == expected_attention_mask


def test_image_processor():
    batch_size = random.choice(range(1, 10))

    config = DTrOCRConfig(lang='en')
    processor = DTrOCRProcessor(config=config)
    tokeniser_output = processor(
        images=[Image.new("RGB", config.image_size[::-1]) for _ in range(batch_size)],
        return_tensors="pt"
    )

    assert tokeniser_output.pixel_values.shape == (batch_size, 3) + tuple(config.image_size)
