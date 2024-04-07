import os
os.environ['HUB_HOME'] = "./modules"
import cv2
import paddlehub as hub

class OcrService():

    def __init__(self):
        self.text = []

    def recognize(self, path):
        np_images =[cv2.imread(path) ] #读取测试文件夹test.txt中的照片路径
        results = ocr.recognize_text(
            images=np_images,         #图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
            use_gpu=False,            #是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
            output_dir='ocr_result',  #图片的保存路径，默认设为 ocr_result；
            visualization=True,       #是否将识别结果保存为图片文件；
            box_thresh=0.5,           #检测文本框置信度的阈值；
            text_thresh=0.5)          #识别中文文本置信度的阈值；

        for result in results:
            data = result['data']
            save_path = result['save_path']
            for infomation in data:
                print('text: ', infomation['text'],
                      '\tconfidence: ', infomation['confidence'],
                      '\tposition: ', infomation['text_box_position'])
                self.text.append(str(infomation['text'] + '\n'))

    def writeText(self, filename):
        with open(filename, 'w') as f:
            for i in self.text:
                f.write(str(i))

if __name__ == '__main__':
    # 加载移动端预训练模型
    # ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    # 服务端可以加载大模型，效果更好
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")

    service = OcrService()
    service.recognize('./img/invincible.jpg')
