import pytesseract as tess
import requests
from PIL import Image, ImageFilter
from io import BytesIO

tess_dir_config = r'--tessdata-dir /usr/share/tesseract-ocr/4.00/tessdata/'
def process_image(url):
    img = _get_image(url)
    img.filter(ImageFilter.SHARPEN)
    res = tess.image_to_string(img, lang='eng', config=tess_dir_config)
    return res

def _get_image(url):
    return Image.open(BytesIO(requests.get(url).content))