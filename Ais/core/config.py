import traceback
import dill as pickle
import os
from datetime import datetime
import sys
import platform

# TODO: .tif files as input

frozen = False
root = os.path.dirname(os.path.dirname(__file__))
app_name = "Ais"
version = "1.0.26"
license = "GNU GPL v3"
log_title = "Ais.log"
log_path = os.path.join(root, log_title)

filetype_segmentation = ".scns"
filetype_traindata = ".scnt"
filetype_semodel = ".scnm"
filetype_semodel_group = ".scnmgroup"


window_width = 1100
window_height = 700
cursor_pos = [0, 0]

error_msg = None
error_new = True
error_obj = None
error_logged = False
error_window_active = False

segmentation_editor = None
se_enabled = True

se_frames = list()
se_active_frame = None
se_models = list()
se_active_model = None
se_path = "..."
se_surface_models = list()

controls_info_text = \
    "left mouse:     draw\n" \
    "    +shift:     place box\n" \
    "right mouse:    erase\n" \
    "    +shift:     erase box\n" \
    "scroll:         change slice\n" \
    "    +shift:     zoom\n" \
    "    + ctrl:     change brush size\n" \
    "spacebar:       reset view\n" \
    "key A:          toggle autocontrast\n" \
    "key I:          toggle interpolation\n" \
    "    +shift:     toggle inversion\n" \
    "key C:          toggle cropping\n" \
    "key F:          toggle flood drawing mode\n" \
    "key Q:          hide 3d models\n" \
    "key O:          toggle overlay visibility\n" \
    "key W:          select previous feature\n" \
    "key S:          select next feature\n" \
    "key left:       previous slice\n" \
    "key right:      next slice\n" \
    "key up:         previous dataset\n" \
    "key down:       next dataset\n"

def set_error(error_object, error_message):
    global error_msg, error_obj, error_new, error_logged
    error_msg = error_message + "\n\n"
    error_msg += "".join(traceback.TracebackException.from_exception(error_object).format())
    print(error_msg)
    error_obj = error_object
    error_new = True
    error_logged = False


def write_to_log(text):
    with open(log_path, "a") as f:
        f.write("\n\n ____________________ \n\n")
        f.write(text)


def start_log():
    with open(log_path, "w") as f:
        f.write(app_name+" version "+version+" "+license+"\n"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+"\n")
        f.write(f"OS: {platform.platform()}")
        f.write(f"Python version: {sys.version}")


def parse_settings():
    sdict = dict()
    with open(os.path.join(root, "core", "settings.txt"), 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            sdict[key] = value
    return sdict


settings = parse_settings()


def edit_setting(key, value):
    global settings
    settings[key] = value
    with open(os.path.join(root, "core", "settings.txt"), 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith(key+"="):
            lines[i] = f"{key}={value}\n"

    with open(os.path.join(root, "core", "settings.txt"), 'w') as f:
        f.writelines(lines)

COLOUR_TEST_A = (1.0, 0.0, 1.0, 1.0)
COLOUR_TEST_B = (0.0, 1.0, 1.0, 1.0)
COLOUR_TEST_C = (1.0, 1.0, 0.0, 1.0)
COLOUR_TEST_D = (1.0, 1.0, 1.0, 1.0)

COLOUR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 0.94)
COLOUR_PANEL_BACKGROUND = (0.94, 0.94, 0.94, 0.94)
COLOUR_TITLE_BACKGROUND = (0.87, 0.87, 0.83, 0.96)
COLOUR_TITLE_BACKGROUND_LIGHT = (0.96, 0.96, 0.93, 0.93)
COLOUR_FRAME_BACKGROUND = (0.87, 0.87, 0.83, 0.96)
COLOUR_FRAME_ACTIVE = (0.91, 0.91, 0.86, 0.94)
COLOUR_FRAME_DARK = (0.83, 0.83, 0.76, 0.94)
COLOUR_FRAME_EXTRA_DARK = (0.76, 0.76, 0.71, 0.94)
COLOUR_MAIN_MENU_BAR = (0.882, 0.882, 0.882, 0.94)
COLOUR_MAIN_MENU_BAR_TEXT = (0.0, 0.0, 0.0, 0.94)
COLOUR_MAIN_MENU_BAR_HILIGHT = (0.96, 0.95, 0.92, 0.94)
COLOUR_MENU_WINDOW_BACKGROUND = (0.96, 0.96, 0.96, 0.94)
COLOUR_DROP_TARGET = COLOUR_FRAME_DARK
COLOUR_HEADER = COLOUR_FRAME_DARK
COLOUR_HEADER_ACTIVE = COLOUR_FRAME_ACTIVE
COLOUR_HEADER_HOVERED = COLOUR_FRAME_EXTRA_DARK
COLOUR_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_TEXT_ACTIVE = (0.0, 0.0, 0.2, 1.0)
COLOUR_TEXT_DISABLED = (0.7, 0.7, 0.7, 1.0)
COLOUR_TEXT_FADE = COLOUR_FRAME_EXTRA_DARK
WINDOW_ROUNDING = 5.0
CONTEXT_MENU_SIZE = (200, 98)
ERROR_WINDOW_HEIGHT = 80
COLOUR_ERROR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 0.94)
COLOUR_ERROR_WINDOW_HEADER = (0.87, 0.87, 0.83, 0.96)
COLOUR_ERROR_WINDOW_HEADER_NEW = (0.87, 0.87, 0.83, 0.96)
COLOUR_ERROR_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_CM_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_CM_OPTION_HOVERED = (1.0, 1.0, 1.0, 1.0)
COLOUR_TRANSPARENT = (1.0, 1.0, 1.0, 0.0)
COLOUR_FRAME_BACKGROUND_BLUE = (0.76, 0.76, 0.83, 1.0)
COLOUR_POSITIVE = (0.1, 0.8, 0.1, 1.0)
COLOUR_NEGATIVE = (0.8, 0.1, 0.1, 1.0)
COLOUR_NEUTRAL = (0.6, 0.6, 0.6, 1.0)
COLOUR_NEUTRAL_LIGHT = (0.8, 0.8, 0.8, 1.0)

TOOLTIP_APPEAR_DELAY = 1.0
TOOLTIP_HOVERED_TIMER = 0.0
TOOLTIP_HOVERED_START_TIME = 0.0

CE_WIDGET_ROUNDING = 50.0