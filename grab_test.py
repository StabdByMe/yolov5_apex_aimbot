import cv2
import numpy as np
import torch
import mss
import win32con
import win32gui
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


def loadModel(device, path):
    model = DetectMultiBackend(path, device=device, dnn=False, data=None, fp16=False)
    model.warmup(imgsz=(1, 3, *[640, 640]))  # warmup
    return model

def foo(device, img):
    # 拿到 dataset 的 im
    im = letterbox(img, [640, 640], stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    # im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im

def inference(device, model, img):
    im = foo(device, img)
    pred = model(im, augment=False, visualize=False)
    # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    det = pred[0]
    annotator = Annotator(img, line_width=3, example=str(names))
    if len(det):
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            hide_labels = False
            hide_conf = False
            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
    return annotator.result()




sct = mss.mss()
def grab(region):
    """
    region: tuple, (left, top, width, height)
    conda install mss / pip install mss
    """
    left, top, width, height = region
    return sct.grab(monitor={'left': left, 'top': top, 'width': width, 'height': height})


windowName = 'Real Time Screen'

# 获取设备, cpu/cuda
device = select_device('0')
# 加载模型
model = loadModel(device, 'runs/trained_models/Apex16/weights/best.pt')

# 网上下的模型
# https://www.youtube.com/watch?v=_QKDEI8uhQQ
# https://github.com/davidhoung2/APEX-yolov5-aim-assist
# model = loadModel(device, 'model.apex.1.pt')

while True:
    # 截图
    img = grab((0, 0, 1920, 1080))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 推测
    img = inference(device, model, img)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 1920, 1080)
    cv2.imshow(windowName, img)

    # 寻找窗口, 设置置顶
    hwnd = win32gui.FindWindow(None, windowName)
    # win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER | win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        cv2.destroyAllWindows()
        exit('ESC ...')
