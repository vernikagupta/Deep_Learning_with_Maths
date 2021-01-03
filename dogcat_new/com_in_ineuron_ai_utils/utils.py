import base64


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)  # converting image into pixel data
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())   # encoding image into base 64 format