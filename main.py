
# coding=utf8
# 环境：
#   python39
#   win10x64
#   pycharm
#   pip3（python39安装包自带）
# 需要安装：
#   PaddleOCR 2.3.0.1
#   paddlepaddle 2.1.2

# 参考：https://blog.csdn.net/zyddj123/article/details/110678758

import os.path
# import paddle.fluid as fluid
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

class OCR:
    def run(self,baseFullPath,resultFullPath):

        ocr = PaddleOCR(use_angle_cls=True, langs="ch")
        result = ocr.ocr(baseFullPath, cls=True)

        image = Image.open(baseFullPath).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        font_path=os.path.realpath('./../PythonOcr/font/pdf.ttf')
        imShow = draw_ocr(image, boxes, txts, scores, 0.5,font_path)
        imShow = Image.fromarray(imShow)
        imShow.save(resultFullPath)

        content = []
        for i in result:
            a = {}
            a['title'] = str(i[1][0])
            a['percent'] = str(i[1][1])
            a['position'] = i[0]
            if a['title']:
                content.append(a)
                print(a['title'])

if __name__ == "__main__":

    baseFullPath = os.path.realpath('./../PythonOcr/img/test.png')      # 上传的文件全路径
    resultFullPath = os.path.realpath('./../PythonOcr/img/test0.png')   # 识别之后的文件全路径

    app = OCR()
    app.run(baseFullPath,resultFullPath)
