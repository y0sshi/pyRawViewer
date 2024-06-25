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
    img_in      = rawio.Raw.new_fromBinFile(path_img_in)
    print(f"raw_in.shape = {img_in.get_shape()}")

    ## initialize RawViewer
    rv = rawviewer.RawViewer('img')
    rv.rawshow(img_in.data, depth=14)

    ## wait keyboard event
    while(True):
        key = cv2.waitKey()

        if   key == ord('q'):
            break
        elif key == ord('p'):
            mouse_image_x, mouse_image_y = rv.get_mouse_coordinate_image()
            print(f"mouse_image(x, y) = ({mouse_image_x}, {mouse_image_y})")
        elif key == ord('r'):
            rv.setrawimage(img_in.data, depth=14)
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

