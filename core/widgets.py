import imgui
from tkinter import filedialog


def toggle_button(label, value, colour=(0.0, 0.64, 0.91, 1.0)):
    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 5))
    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
    imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 10)
    imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 15)
    pop_colour = False
    if value:
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *colour)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *colour)
        pop_colour = True
    imgui.align_text_to_frame_padding()
    imgui.text(label)
    imgui.same_line()
    imgui.set_next_item_width(40)
    changed, value = imgui.slider_int(f"##_{label}", value, 0, 1, "")
    if pop_colour:
        imgui.pop_style_color(2)
    imgui.pop_style_var(4)
    return changed, value


def centred_button(label, width, height, rounding=10):
    cw = imgui.get_content_region_available_width()
    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, rounding)
    imgui.new_line()
    imgui.same_line(spacing=(cw - width) / 2)
    retval = False
    if imgui.button(label, width, height):
        retval = True
    imgui.pop_style_var(1)
    return retval

def select_directory(label, path):
    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (3, 3))
    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 0)
    cw = imgui.get_content_region_available_width()
    imgui.set_next_item_width(cw - 65)
    changed, path = imgui.input_text(f"##_{label}", path, 256)
    imgui.same_line()
    if imgui.button(label, 55, 19):
        selected_dir = filedialog.askdirectory()
        if selected_dir is not None and selected_dir != "":
            path = selected_dir
    imgui.pop_style_var(2)
    return changed, path