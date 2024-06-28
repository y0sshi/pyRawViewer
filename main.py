''' main.py '''
__version__ = "0.0.1"
# ======================================================================
# Import
# ======================================================================
import sys
import cv2
import rawviewer
import rawio


# ======================================================================
# Functions
# ======================================================================
def main():
    ## arguments
    path_img_in = sys.argv[1]
    img_in      = rawio.Raw.new_fromRawFile(path_img_in)
    print(f"raw_in.shape = {img_in.get_shape()}")

    ## initialize RawViewer
    rv = rawviewer.RawViewer('img')
    rv.rawshow(img_in.data, bitwidth=14)

    ## wait keyboard event
    while(True):
        key = cv2.waitKey()

        if   key == ord('q'):
            break
        elif key == ord('i'):
            scale_x, scale_y = rv.get_scale()
            print(f"scale_x = {scale_x}, scale_y = {scale_y}")
            print(f"gain = {rv.get_gain()}")
        elif key == ord('c'):
            mouse_image_x, mouse_image_y = rv.get_mouse_coordinate_image()
            print(f"mouse_image(x, y) = ({mouse_image_x}, {mouse_image_y})")
        elif key == ord('r'):
            rv.setrawimage(img_in.data, bitwidth=14)
            rv.redraw_image()
        elif key == ord('p'):
            pedestal = int(input("pedestal="))
            rv.set_pedestal(pedestal=pedestal)
            rv.redraw_image()
        elif key == ord('g'):
            gamma = float(input("gamma="))
            rv.set_gamma(gamma=gamma)
            rv.redraw_image()
        elif key == ord('>'):
            rv.set_gain(rv.get_gain() * 2.0)
            rv.redraw_image()
        elif key == ord('<'):
            rv.set_gain(rv.get_gain() * 0.5)
            rv.redraw_image()
        else:
            pass

    cv2.destroyAllWindows()

    return


# ======================================================================
# Scripts
# ======================================================================
if __name__ == "__main__":
    main()

