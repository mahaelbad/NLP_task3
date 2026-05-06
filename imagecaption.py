from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def generate_caption(image):

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    inputs = processor(
        image,
        return_tensors="pt"
    )

    output = model.generate(**inputs, max_length=30, num_beams=5)

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption