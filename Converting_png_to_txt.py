
#converting images to txt files


from pytesseract import image_to_string
from PIL import Image
import os

tessdata_dir_config = r'--tessdata-dir "/usr/local/Cellar/tesseract/4.1.1/share/tessdata"'

path = os.chdir(os.path.join(os.getcwd()) + "")

for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        print(filename)
        txt = image_to_string(filename, config=tessdata_dir_config)
        with open(filename.replace(".jpg", ".txt"), mode='w') as f:
            f.write(txt)








