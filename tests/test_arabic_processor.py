from dtrocr.config import DTrOCRConfig
from dtrocr.processor import DTrOCRProcessor

import random
from PIL import Image


def test_tokeniser_preprocess():
    tokeniser = DTrOCRProcessor(config=DTrOCRConfig(lang='ar'), add_bos_token=True)
    tokeniser_output = tokeniser(texts=["جُمْلَةٌ لاِخْتِبَارِ أَرَا جِي بِي تِي", "جملة لاختبار ارا جي بي تي"])

    assert tokeniser_output.input_ids[0] == tokeniser_output.input_ids[1]
    assert tokeniser_output.attention_mask[0] == tokeniser_output.attention_mask[1]


def test_tokeniser_with_bos_token():
    tokeniser = DTrOCRProcessor(config=DTrOCRConfig(lang='ar'), add_bos_token=True)
    tokeniser_output = tokeniser(texts=["هذه جملة", "هذه ليست جملة، آسف"])
    expected_input_ids = [
        [280, 1043, 8162],
        [280, 1043, 2434, 8162, 297, 49612]
    ]
    expected_attention_mask = [
        [1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ]

    assert tokeniser_output.input_ids == expected_input_ids
    assert tokeniser_output.attention_mask == expected_attention_mask


def test_tokeniser_with_eos_token():
    tokeniser = DTrOCRProcessor(config=DTrOCRConfig(lang='ar'), add_eos_token=True)
    tokeniser_output = tokeniser(texts=["هذه جملة", "هذه ليست جملة، آسف"])

    expected_input_ids = [
        [280, 1043, 8162, 63999],
        [280, 1043, 2434, 8162, 297, 49612, 63999]
    ]
    expected_attention_mask = [
        [1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]

    assert tokeniser_output.input_ids == expected_input_ids
    assert tokeniser_output.attention_mask == expected_attention_mask


def test_tokeniser_with_eos_and_bos_tokens():
    tokeniser = DTrOCRProcessor(config=DTrOCRConfig(lang='ar'), add_bos_token=True, add_eos_token=True)
    tokeniser_output = tokeniser(texts=["هذه جملة", "هذه ليست جملة، آسف"])

    expected_input_ids = [
        [63999, 280, 1043, 8162, 63999],
        [63999, 280, 1043, 2434, 8162, 297, 49612, 63999]
    ]
    expected_attention_mask = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]

    assert tokeniser_output.input_ids == expected_input_ids
    assert tokeniser_output.attention_mask == expected_attention_mask


def test_image_processor():
    batch_size = random.choice(range(1, 10))

    config = DTrOCRConfig(lang='ar')
    processor = DTrOCRProcessor(config=config)
    tokeniser_output = processor(
        images=[Image.new("RGB", config.image_size[::-1]) for _ in range(batch_size)],
        return_tensors="pt"
    )

    assert tokeniser_output.pixel_values.shape == (batch_size, 3) + tuple(config.image_size)
