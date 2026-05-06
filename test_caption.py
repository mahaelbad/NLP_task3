from imagecaption import generate_caption
from PIL import Image

image = Image.open("test.jpg")

caption = generate_caption(image)

print(caption)