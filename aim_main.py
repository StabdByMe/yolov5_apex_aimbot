import threading
import time
import tkinter
import pynput
from aimbot import Aimbot
import mouse_control



ExitFlag = False  # 退出程序标记
SwitchShowBox = True  # 是否显框
SwitchAutoAim = False  # 是否自瞄


def onClick(x, y, button, pressed):
    # print(f'button {button} {"pressed" if pressed else "released"} at ({x},{y})')
    global ExitFlag, SwitchShowBox, SwitchAutoAim
    if ExitFlag:
        return False  # 结束监听线程
    if not pressed:
        if pynput.mouse.Button.x2 == button:
            # 侧上键
            # SwitchShowBox = not SwitchShowBox
            print(f'Switch ShowBox: {"enable" if SwitchShowBox else "disable"}')
        elif pynput.mouse.Button.x1 == button:
            # 侧下键
            SwitchAutoAim = not SwitchAutoAim
            print(f'Switch AutoAim: {"enable" if SwitchAutoAim else "disable"}')


def onRelease(key):
    # print(f'{key} released')
    if key == pynput.keyboard.Key.end:
        global ExitFlag
        ExitFlag = True
        return False


mouseListener = pynput.mouse.Listener(on_click=onClick)
mouseListener.start()
keyboardListener = pynput.keyboard.Listener(on_release=onRelease)
keyboardListener.start()
mouse = pynput.mouse.Controller()


def draw(canvas, x1, y1, x2, y2, width=2, color='red', text=None):
    canvas.create_rectangle(x1, y1, x2, y2, width=width, outline=color)
    if text is not None:
        canvas.create_rectangle(x1, y1 - 20, x2, y1, fill='black')
        canvas.create_text(x1, y1, anchor='sw', text=text, fill='yellow', font=('', 16))


# 主程序
TRANSCOLOUR = 'gray'
root = tkinter.Tk()  # 创建
root.attributes('-fullscreen', 1)  # 全屏
root.attributes('-topmost', -1)  # 置顶
root.wm_attributes('-transparentcolor', TRANSCOLOUR)  # 设置透明且穿透的颜色
root['bg'] = TRANSCOLOUR  # 设置透明且穿透
# 添加画布
canvas = tkinter.Canvas(root, background=TRANSCOLOUR, borderwidth=0, highlightthickness=0)
canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
canvas.pack()

aimbot = Aimbot([640, 360, 640, 360], 'runs/trained_models/Apex/best.pt')
# aimbot = Aimbot([0, 0, 3440, 1440], '../yolov5s.pt')
# aimbot = Aimbot([0, 0, 3440, 1440], '../yolov5n6.pt')
# aimbot = Aimbot([3440 // 3, 1440 // 3, 3440 // 3, 1440 // 3], '../yolov5s.pt')
print("加载完成")


def foo():
    global ExitFlag, SwitchShowBox, SwitchAutoAim
    while ExitFlag is False:

        # print(f'{SwitchShowBox}, {SwitchAutoAim}')
        if (SwitchShowBox is False) & (SwitchAutoAim is False):
            continue

        t1 = time.perf_counter()
        canvas.delete(tkinter.ALL)
        t2 = time.perf_counter()
        aims = aimbot.getAims()
        t3 = time.perf_counter()
        for aim in aims:
            if SwitchShowBox:
                draw(canvas, aim[1], aim[2], aim[1] + aim[3], aim[2] + aim[4], 5, text=aim[0])
                time.sleep(0.05)
                if SwitchAutoAim:
                    print('开锁！')
                    # 瞄准, 预留
                    t4 = time.perf_counter()
                    mouse_control.lock(aim, mouse)
                    t5 = time.perf_counter()
                    print(f'瞄准用时:{int((t5 - t4) * 1000)}ms')

        # print(f'画布清理:{int((t2 - t1) * 1000)}ms, 目标检测:{int((t3 - t2) * 1000)}ms, 目标数量:{len(aims)}, 画框:{int((t4 - t3) * 1000)}ms, 瞄准:{int((t5 - t4) * 1000)}ms, 总计:{int((t5 - t1) * 1000)}ms, 画框开关:{SwitchShowBox}, 自瞄开关:{SwitchAutoAim}')

    # 循环结束, 程序结束
    canvas.delete(tkinter.ALL)
    # 关闭主窗口来结束主线程
    print('Esc')
    root.destroy()


# 创建并启动线程
t = threading.Thread(target=foo, daemon=True)
t.start()

# 主循环
root.mainloop()
