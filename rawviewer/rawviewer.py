''' rawviewer.py '''
__version__ = "0.0.1"
# ======================================================================
# Import
# ======================================================================
import math
import cv2
import numpy as np

from rawviewer import affine
from rawviewer import colorfilter


# ======================================================================
# Classes
# ======================================================================
class RawViewer:
    def __init__(self, winname: str, back_color = (128, 128, 128), inter = cv2.INTER_NEAREST):
        '''Instantiate the RawViewer Class.

        Parameters
        ----------
        winname : str
            Name of the window
        back_color : tuple, optional
            background color, by default (128, 128, 128)
        inter : optional
            interpolation meghods, by default cv2.INTER_NEAREST

        Returns
        -------
        '''
        self.__winname                   = winname
        self.__back_color                = back_color
        self.__inter                     = inter

        self.__src_image                 = None
        self.__src_gained_image          = None
        self.__value_image               = None
        self.__disp_image                = None
        self.__affine_matrix             = affine.identityMatrix(affine.AFFINE_MATRIX_SIZE)
        self.__old_affine_matrix         = affine.identityMatrix(affine.AFFINE_MATRIX_SIZE)

        self.__gamma                     = 2.2

        self.__zoom_delta                = 2.0
        self.__min_scale                 = 0.01
        self.__max_scale                 = 300

        self.__bright_disp_enabled       = True
        self.__min_bright_disp_scale     = 30
        self.__grid_disp_enabled         = True
        self.__grid_color                = (128, 128, 128)
        self.__min_grid_disp_scale       = 20

        self.__gain                      = 1.0
        self.__bitwidth                  = 10
        self.__pedestal                  = 64

        self.__color_disp_enable         = True

        self.__mouse_event_enabled       = True
        self.__mouse_down_flag           = False
        self.__mouse_coordinate_window_x = 0
        self.__mouse_coordinate_window_y = 0

        self.__color_filter              = colorfilter.CF_QUAD

        self.__mouse_callback_func       = None

        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

        cv2.setMouseCallback(winname, self._onMouse, winname)


    def imshow(self, image, zoom_fit :bool = True):
        '''Image display
        '''
        if image is None:
            print('imshow error')
            return
    
        self.__src_image = image
        if zoom_fit is True:
            self.redraw_image()
            self.zoom_fit()
        else:
            self.redraw_image()


    def rawshow(self, image, bitwidth :int = 10, zoom_fit :bool = True):
        '''Image display
        '''
        self.__bitwidth = bitwidth

        if image is None:
            print('rawshow error')
            return
    
        self.setrawimage(image, self.__gain, self.__bitwidth, self.__gamma, self.__pedestal)

        if zoom_fit is True:
            self.redraw_image()
            self.zoom_fit()
        else:
            self.redraw_image()


    def set_pedestal(self, pedestal):
        self.__pedestal = pedestal
        self.setrawimage(self.__value_image, self.__gain, self.__bitwidth, self.__gamma, self.__pedestal)


    def set_gain(self, gain):
        self.__gain = gain
        self.setrawimage(self.__value_image, self.__gain, self.__bitwidth, self.__gamma, self.__pedestal)


    def set_gamma(self, gamma):
        self.__gamma = gamma
        self.setrawimage(self.__value_image, self.__gain, self.__bitwidth, self.__gamma, self.__pedestal)


    def get_gain(self):
        return self.__gain


    def get_scale(self):
        scale_x = self.__affine_matrix[0, 0]
        scale_y = self.__affine_matrix[1, 1]

        return scale_x, scale_y


    def get_mouse_coordinate_window(self):
        mouse_x = int(self.__mouse_coordinate_window_x + 0.5)
        mouse_y = int(self.__mouse_coordinate_window_y + 0.5)
        return mouse_x, mouse_y


    def get_mouse_coordinate_image(self):
        mouse_x_float, mouse_y_float = self.convert_coordinate_from_window_to_image(self.__mouse_coordinate_window_x, self.__mouse_coordinate_window_y)
        mouse_x = int(mouse_x_float + 0.5)
        mouse_y = int(mouse_y_float + 0.5)
        return mouse_x, mouse_y


    def setrawimage(self, image, gain = 1.0, bitwidth :int = 10, gamma = 2.2, pedestal = 64):
        if image.ndim == 2:
            self.__src_image = self._convert_showimg(image, bitwidth, gamma, pedestal)
        else:
            self.__src_image = image

        self.__src_gained_image = np.clip(self.__src_image * gain, 0.0, 255).astype(np.uint8)
        self.__value_image      = image


    def _convert_showimg(self, rawimg, bitwidth: int = 10, gamma = 2.2, pedestal = 64):
        max_val            = 2 ** bitwidth - 1 - pedestal
        rawimg_wo_pedestal = np.clip(rawimg.astype(np.int32) - pedestal, 0, max_val)
        rawimg_f64         = rawimg_wo_pedestal.astype(np.float64)
        tmp                = 255 * ((rawimg_f64 / max_val) ** (1 / gamma))
        showimg            = (np.clip(tmp, 0.0, max_val) + 0.5).astype(np.uint8)
        if showimg.ndim == 2:
            return np.stack([showimg, showimg, showimg], 2)
        else:
            return showimg



    def _onMouse(self, event, x, y, flags, params):
        '''Mouse-event Callback function

        Parameters
        ----------
        event : int
            Mouse-event ID
        x : int
            coordinate-x of Mouse position
        y : int
            coordinate-y of Mouse position
        flags : int
            key flags (<Shift>, <Ctrl>, <Alt>)
        params : int
            parameters of callback

        Returns
        -------
        '''
        if self.__mouse_event_enabled is False:
            return

        if self.__disp_image is None:
            return

        if self.__mouse_callback_func is not None:
            x_img, y_img = self.convert_coordinate_from_window_to_image(x, y)
            self._callback_handler(self.__mouse_callback_func, self, event, x, y, flags, params, x_img, y_img, self.__affine_matrix[0, 0])

        if event == cv2.EVENT_LBUTTONDOWN:
            ## when left clicked-down
            self.__mouse_down_flag   = True
            self.__old_affine_matrix = self.__affine_matrix
            self.old_point_x = x
            self.old_point_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            ## when left clicked-up
            self.__mouse_down_flag   = False

        elif event == cv2.EVENT_MOUSEMOVE:
            ## when mousemove
            self.__mouse_coordinate_window_x = x
            self.__mouse_coordinate_window_y = y

            if self.__mouse_down_flag is True:
                self.__affine_matrix = affine.translateMatrix(x - self.old_point_x, y - self.old_point_y).dot(self.__old_affine_matrix)
                self.redraw_image()

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom_at(self.__zoom_delta, x, y)
            else:
                self.zoom_at(1/self.__zoom_delta, x, y)
            #print(f"scale = {scale}")

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            print("zoom_fit()")
            self.zoom_fit()

        elif event == cv2.EVENT_RBUTTONDBLCLK:
            scale = 1/self.__affine_matrix[0, 0]
            self.__affine_matrix = affine.scaleAtMatrix(scale, scale, x, y).dot(self.__affine_matrix)


    def _callback_handler(self, func, *args):
        return func(*args)


    def zoom_at(self, delta: float, wx: float, wy: float):
        if   delta >= 1.0:
            if self.__affine_matrix[0, 0] * delta > self.__max_scale:
                return
            self.__affine_matrix = affine.scaleAtMatrix(delta, delta, wx, wy).dot(self.__affine_matrix)
        elif delta >= 0.0:
            if self.__affine_matrix[0, 0] * delta < self.__min_scale:
                return
            self.__affine_matrix = affine.scaleAtMatrix(delta, delta, wx, wy).dot(self.__affine_matrix)
        else:
            return

        self.redraw_image()


    def zoom_fit(self, image_width: int = 0, image_height: int = 0):
        if self.__src_image is not None:
            image_width  = self.__src_image.shape[1]
            image_height = self.__src_image.shape[0]
        else:
            if image_width == 0 or image_height == 0:
                return

        try:
            _, _, win_width, win_height = cv2.getWindowImageRect(self.__winname)
        except:
            print('zoom fit error')
            return

        if (image_width * image_height <= 0) or (win_width * win_height <= 0):
            print(f'zoom fit error : image(w, h) = ({image_width}, {image_height}), win(w, h) = ({win_width}, {win_height})')
            return

        self.__affine_matrix = affine.identityMatrix(affine.AFFINE_MATRIX_SIZE)
        scale   = 1.0
        offsetx = 0.0
        offsety = 0.0
        if (win_width * image_height) > (image_width * win_height):
            scale   = win_height / image_height
            offsetx = (win_width - image_width * scale) / 2.0
        else:
            scale   = win_width / image_width
            offsety = (win_height - image_height * scale) / 2.0

        self.__affine_matrix = affine.translateMatrix(0.5, 0.5).dot(self.__affine_matrix)
        self.__affine_matrix = affine.scaleMatrix(scale, scale).dot(self.__affine_matrix)
        self.__affine_matrix = affine.translateMatrix(offsetx, offsety).dot(self.__affine_matrix)

        self.redraw_image()


    def redraw_image(self):
        '''image redraw
        '''
        if self.__src_image is None:
            print('redraw_image() : src_image none')
            return

        try:
            _, _, win_width, win_height = cv2.getWindowImageRect(self.__winname)
        except:
            print('redraw image error')
            return

        ## colored display
        if self.__color_disp_enable is True:
            tile_repeats = ((len(self.__src_gained_image)    // len(self.__color_filter)),     ## height
                            (len(self.__src_gained_image[0]) // len(self.__color_filter[0])))  ## width

            src_image = self.__src_gained_image[:,:,0]
            disp_image_b = np.where(np.tile(np.equal(self.__color_filter, "B") |
                                            np.equal(self.__color_filter, "C") |
                                            np.equal(self.__color_filter, "M") |
                                            np.equal(self.__color_filter, "W"), tile_repeats), src_image, 0)

            disp_image_g = np.where(np.tile(np.equal(self.__color_filter, "GR") |
                                            np.equal(self.__color_filter, "GB") |
                                            np.equal(self.__color_filter, "C")  |
                                            np.equal(self.__color_filter, "Y")  |
                                            np.equal(self.__color_filter, "W"), tile_repeats), src_image, 0)

            disp_image_r = np.where(np.tile(np.equal(self.__color_filter, "R") |
                                            np.equal(self.__color_filter, "M")  |
                                            np.equal(self.__color_filter, "Y")  |
                                            np.equal(self.__color_filter, "W"), tile_repeats), src_image, 0)

            colored_src_image = np.stack([disp_image_b, disp_image_g, disp_image_r], axis=2)
            self.__disp_image = cv2.warpAffine(colored_src_image, self.__affine_matrix[:2,], (win_width, win_height), flags = self.__inter, borderValue = self.__back_color)
        else:
            self.__disp_image = cv2.warpAffine(self.__src_gained_image, self.__affine_matrix[:2,], (win_width, win_height), flags = self.__inter, borderValue = self.__back_color)


        ## draw grid-line
        if self.__grid_disp_enabled is True:
            if self.__affine_matrix[0, 0] > self.__min_grid_disp_scale:
                self._draw_grid_line()


        ## display bright-value
        if self.__bright_disp_enabled is True:
            if self.__affine_matrix[0, 0] > self.__min_bright_disp_scale:
                self._draw_bright_value()


        ## show image
        cv2.imshow(self.__winname, self.__disp_image)


    def _draw_grid_line(self):
        ret, x_begin, y_begin, x_end, y_end = self._image_disp_rect()
        if ret is False:
            return
        
        mat = self.__affine_matrix
        grid_phase_height, grid_phase_width = self.__color_filter.shape

        ## vertical line
        py_begin = math.floor(mat[1, 1] * y_begin + mat[1, 2] + 0.5)
        py_end   = math.floor(mat[1, 1] * y_end   + mat[1, 2] + 0.5)
        for x in range(int(x_begin + 0.5), int(x_end + 0.5)):
            px = mat[0, 0] * (x - 0.5) + mat[0, 2]
            if   (int(x % grid_phase_width) == 0):
                cv2.line(self.__disp_image,
                         (math.floor(px + 0.5), py_begin),
                         (math.floor(px + 0.5), py_end  ),
                         (128, 128, 128),
                         2)
            else:
                cv2.line(self.__disp_image,
                         (math.floor(px + 0.5), py_begin),
                         (math.floor(px + 0.5), py_end  ),
                         self.__grid_color,
                         1)

        ## horizontal line
        px_begin = math.floor(mat[0, 0] * x_begin + mat[0, 2] + 0.5)
        px_end   = math.floor(mat[0, 0] * x_end   + mat[0, 2] + 0.5)
        for y in range(int(y_begin + 0.5), int(y_end + 0.5)):
            py = mat[1, 1] * (y - 0.5) + mat[1, 2]
            if   (int(y % grid_phase_height) == 0):
                cv2.line(self.__disp_image,
                         (px_begin, math.floor(py + 0.5)),
                         (px_end  , math.floor(py + 0.5)),
                         (128, 128, 128),
                         2)
            else:
                cv2.line(self.__disp_image,
                         (px_begin, math.floor(py + 0.5)),
                         (px_end  , math.floor(py + 0.5)),
                         self.__grid_color,
                         1)

    def _draw_bright_value(self):
        ret, x_begin, y_begin, x_end, y_end = self._image_disp_rect()
        if ret is False:
            return
        mat = self.__affine_matrix

        py_begin = math.floor(mat[1, 1] * y_begin + mat[1, 2] + 0.5)
        py_end   = math.floor(mat[1, 1] * y_end   + mat[1, 2] + 0.5)
        px_begin = math.floor(mat[0, 0] * x_begin + mat[0, 2] + 0.5)
        px_end   = math.floor(mat[0, 0] * x_end   + mat[0, 2] + 0.5)

        offset_x   = int(mat[0, 0] / 90)
        offset_y   = int(mat[0, 0] /  6)

        offset_y_r = int(mat[0, 0] / 1.58)
        offset_y_g = int(mat[0, 0] / 1.24)
        offset_y_b = int(mat[0, 0] / 1.03)

        fore_r   = (  0,   0, 200)
        fore_g   = (  0, 200,   0)
        fore_b   = (200,   0,   0)
        fore_c   = (200, 200,   0)
        fore_m   = (200,   0, 200)
        fore_y   = (  0, 200, 200)

        if mat[0, 0] > 100:
            thick = 2
        else:
            thick = 1

        if self.__disp_image.ndim == 3 and self.__value_image.ndim == 3:
            ## RGB-image
            for y in range(int(y_begin + 0.5), int(y_end + 0.5)):
                for x in range(int(x_begin + 0.5), int(x_end + 0.5)):
                    px = mat[0, 0] * (x - 0.5) + mat[0, 2]
                    py = mat[1, 1] * (y - 0.5) + mat[1, 2]

                    bright = self.__src_image[y, x, :]

                    if bright.max() > 127:
                        fore_color = (  0,   0,   0)
                    else:
                        fore_color = (255, 255, 255)

                    ## coodrinate
                    cv2.putText(self.__disp_image,
                                text      = f"({x}, {y})",
                                org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y),
                                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = mat[0, 0] / 200,
                                color     = fore_color,
                                thickness = thick)

                    ## B
                    cv2.putText(self.__disp_image,
                                text      = f"{str(bright[0]).rjust(11)}",
                                org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_b),
                                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = mat[0, 0] / 200,
                                color     = fore_b,
                                thickness = thick)

                    ## G
                    cv2.putText(self.__disp_image,
                                text      = f"{str(bright[1]).rjust(11)}",
                                org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_g),
                                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = mat[0, 0] / 200,
                                color     = fore_g,
                                thickness = thick)

                    ## R
                    cv2.putText(self.__disp_image,
                                text      = f"{str(bright[2]).rjust(11)}",
                                org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_r),
                                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = mat[0, 0] / 200,
                                color     = fore_r,
                                thickness = thick)

        elif self.__disp_image.ndim == 3 and self.__value_image.ndim == 2:
            ## raw-image
            for y in range(int(y_begin + 0.5), int(y_end + 0.5)):
                for x in range(int(x_begin + 0.5), int(x_end + 0.5)):
                    px = mat[0, 0] * (x - 0.5) + mat[0, 2]
                    py = mat[1, 1] * (y - 0.5) + mat[1, 2]

                    bright = self.__src_image[y, x, :]

                    if bright.max() > 127:
                        fore_color = (  0,   0,   0)
                    else:
                        fore_color = (255, 255, 255)

                    cid = self._get_cid(x, y)

                    ## coodrinate
                    cv2.putText(self.__disp_image,
                                text      = f"({x}, {y})",
                                org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y),
                                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = mat[0, 0] / 200,
                                color     = fore_color,
                                thickness = thick)

                    if cid == "B":
                        cv2.putText(self.__disp_image,
                                    text      = f"{str(self.__value_image[y, x]).rjust(11)}",
                                    org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_b),
                                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = mat[0, 0] / 200,
                                    color     = fore_b,
                                    thickness = thick)

                    if cid == "GR" or cid == "GB":
                        cv2.putText(self.__disp_image,
                                    text      = f"{str(self.__value_image[y, x]).rjust(11)}",
                                    org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_b),
                                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = mat[0, 0] / 200,
                                    color     = fore_g,
                                    thickness = thick)

                    if cid == "R":
                        cv2.putText(self.__disp_image,
                                    text      = f"{str(self.__value_image[y, x]).rjust(11)}",
                                    org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_b),
                                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = mat[0, 0] / 200,
                                    color     = fore_r,
                                    thickness = thick)

                    if cid == "C":
                        cv2.putText(self.__disp_image,
                                    text      = f"{str(self.__value_image[y, x]).rjust(11)}",
                                    org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_b),
                                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = mat[0, 0] / 200,
                                    color     = fore_c,
                                    thickness = thick)

                    if cid == "M":
                        cv2.putText(self.__disp_image,
                                    text      = f"{str(self.__value_image[y, x]).rjust(11)}",
                                    org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_b),
                                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = mat[0, 0] / 200,
                                    color     = fore_m,
                                    thickness = thick)

                    if cid == "Y":
                        cv2.putText(self.__disp_image,
                                    text      = f"{str(self.__value_image[y, x]).rjust(11)}",
                                    org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_b),
                                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = mat[0, 0] / 200,
                                    color     = fore_y,
                                    thickness = thick)

                    if cid == "W":
                        cv2.putText(self.__disp_image,
                                    text      = f"{str(self.__value_image[y, x]).rjust(11)}",
                                    org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_b),
                                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = mat[0, 0] / 200,
                                    color     = fore_color,
                                    thickness = thick)

        else:
            ## mono-image
            for y in range(int(y_begin + 0.5), int(y_end + 0.5)):
                for x in range(int(x_begin + 0.5), int(x_end + 0.5)):
                    px = mat[0, 0] * (x - 0.5) + mat[0, 2]
                    py = mat[1, 1] * (y - 0.5) + mat[1, 2]

                    bright = self.__src_image[y, x, :]

                    if bright.max() > 127:
                        fore_color = (  0,   0,   0)
                    else:
                        fore_color = (255, 255, 255)

                    ## coodrinate
                    cv2.putText(self.__disp_image,
                                text      = f"({x}, {y})",
                                org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y),
                                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = mat[0, 0] / 200,
                                color     = fore_color,
                                thickness = thick)

                    cv2.putText(self.__disp_image,
                                text      = f"{str(self.__value_image[y, x]).rjust(11)}",
                                org       = (math.floor(px + 0.5) + offset_x, math.floor(py + 0.5) + offset_y_b),
                                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = mat[0, 0] / 200,
                                color     = fore_color,
                                thickness = thick)


    def _image_disp_rect(self):
        '''args
        '''

        if self.__src_image is None:
            print('_image_disp_rect error')
            return False, 0, 0, 0, 0

        coordinate_window_top_left     = self.convert_coordinate_from_image_to_window(-0.5, -0.5)
        image_width  = self.__src_image.shape[1]
        image_height = self.__src_image.shape[0]
        coordinate_window_bottom_right = self.convert_coordinate_from_image_to_window(image_width - 0.5, image_height - 0.5)

        try:
            _, _, win_width, win_height = cv2.getWindowImageRect(self.__winname)
        except:
            print('_image_disp_rect error')
            return False, 0, 0, 0, 0

        coordinate_img_top_left     = self.convert_coordinate_from_window_to_image(-0.5, -0.5)
        coordinate_img_bottom_right = self.convert_coordinate_from_window_to_image(win_width - 0.5, win_height - 0.5)
        
        invMat = affine.inverse(self.__affine_matrix)

        if coordinate_window_top_left[0] < 0:
            image_left = math.floor(invMat[0, 2] + 0.5) - 0.5
        else:
            image_left = -0.5
        if coordinate_window_top_left[1] < 0:
            image_top = math.floor(invMat[1, 2] + 0.5) - 0.5
        else:
            image_top = -0.5
        if coordinate_window_bottom_right[0] > win_width - 1:
            image_right = math.floor(invMat[0, 0] * (win_width  - 1) + invMat[0, 2] + 0.5) - 0.5
            pass
        else:
            image_right = image_width - 0.5
            pass
        if coordinate_window_bottom_right[1] > win_height - 1:
            image_bottom = math.floor(invMat[1, 1] * (win_height - 1) + invMat[1, 2] + 0.5) - 0.5
        else:
            image_bottom = image_height - 0.5

        return True, image_left, image_top, image_right, image_bottom


    def convert_coordinate_from_image_to_window(self, x_img: float, y_img: float):
        x_window, y_window = affine.affinePoint(self.__affine_matrix, x_img, y_img)
        return x_window, y_window


    def convert_coordinate_from_window_to_image(self, x_window: float, y_window: float):
        invMat = affine.inverse(self.__affine_matrix)
        x_img, y_img = affine.affinePoint(invMat, x_window, y_window)
        return x_img, y_img


    def _get_cid(self, x, y):
        cfa_height, cfa_width = self.__color_filter.shape
        return self.__color_filter[y % cfa_height, x % cfa_width]
        
