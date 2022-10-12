import time

import cv2
import numpy as np
import torch
import mss
import win32con
import win32gui
import win32api
import win32print

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


# 获取缩放后的分辨率
def getScreenResolution():
    return win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN), win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)


# 获取真实的分辨率
def getScreenRealResolution():
    hDC = win32gui.GetDC(0)
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)  # 横向分辨率
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)  # 纵向分辨率
    return w, h


sct = mss.mss()


def grab(monitor):
    img = sct.grab(monitor=monitor)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def getDrivce():
    return select_device('');


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


class Aimbot():
    def __init__(self, grab, model):
        # 屏幕宽高
        resolution = getScreenResolution()
        self.sw = resolution[0]
        self.sh = resolution[1]
        # 截屏范围 grab = [left, top, width, height]
        self.gl = grab[0]
        self.gt = grab[1]
        self.gw = grab[2]
        self.gh = grab[3]
        self.grab = {'left': self.gl, 'top': self.gt, 'width': self.gw, 'height': self.gh}
        # yolo
        self.device = getDrivce()
        self.model = loadModel(self.device, model)

    def getAims(self):


        # 截图
        t1 = time.perf_counter()
        img = grab(self.grab)
        t2 = time.perf_counter()
        # 检测
        im = foo(self.device, img)

        pred = self.model(im, augment=False, visualize=False)
        # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        det = pred[0]
        aims = []
        if len(det):
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                c = int(cls)  # integer class

                label = f'{names[c]} {conf:.2f}'
                # 计算相对屏幕坐标系的点位
                left = self.gl + ((xywh[0] * self.gw) - (xywh[2] * self.gw) / 2)
                top = self.gt + ((xywh[1] * self.gh) - (xywh[3] * self.gh) / 2)
                width = xywh[2] * self.gw
                height = xywh[3] * self.gh

                aims.append([label, left, top, width, height])
                #print(str(left) + '  ' + str(top))
        t3 = time.perf_counter()
        print(f'截图:{int((t2 - t1) * 1000)}ms, 目标检测:{int((t3 - t2) * 1000)}ms, 目标数量:{len(aims)}, 总计:{int((t3 - t1) * 1000)}ms')
        return aims
