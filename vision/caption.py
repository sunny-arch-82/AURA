from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model once (outside function)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(path):
    """Generate a caption for an image file."""
    raw_image = Image.open(path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
