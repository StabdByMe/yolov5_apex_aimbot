import pynput
import time
import pydirectinput
import win32api
import win32con


def lock(aim, mouse):
    # x, y 对应分辨率宽、高，mouse 为鼠标对象
    mouse_pos_x, mouse_pos_y = mouse.position
    #print(mouse_pos_x, mouse_pos_y)
    print('目标位置：' + 'x=' + str(aim[1] + aim[3] / 2) + ',y=' + str(aim[2] + aim[4] / 3))
    #pydirectinput.move(round((aim[1] + aim[3] / 2) - mouse_pos_x), round((aim[2] + aim[4] / 3) - mouse_pos_y))
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, round((aim[1] + aim[3] / 2) - mouse_pos_x), round((aim[2] + aim[4] / 3) - mouse_pos_y))
    print('锁住！')



