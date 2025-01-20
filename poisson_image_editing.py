import argparse
import numpy as np
import cv2
from os import path
import scipy.sparse
from scipy.sparse.linalg import spsolve
import getopt
import sys

class MaskPainter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image_copy = self.image.copy()
        self.mask = np.zeros(self.image.shape)
        self.mask_copy = self.mask.copy()
        self.size = 4
        self.to_draw = False
        self.window_name = "Draw mask. s:save; r:reset; q:quit"

    def _paint_mask_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.to_draw = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.to_draw:
                cv2.rectangle(self.image, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 255, 0), -1)
                cv2.rectangle(self.mask, (x-self.size, y-self.size), (x+self.size, y+self.size), (255, 255, 255), -1)
                cv2.imshow(self.window_name, self.image)
        elif event == cv2.EVENT_LBUTTONUP:
            self.to_draw = False

    def paint_mask(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._paint_mask_handler)
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.image = self.image_copy.copy()
                self.mask = self.mask_copy.copy()
            elif key == ord("s"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()
        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)
        mask_path = path.join(path.dirname(self.image_path), 'mask.png')
        cv2.imwrite(mask_path, self.mask)
        cv2.destroyAllWindows()
        return mask_path

class MaskMover:
    def __init__(self, image_path, mask_path):
        self.image_path, self.mask_path = image_path, mask_path
        self.image = cv2.imread(image_path)
        self.image_copy = self.image.copy()
        self.original_mask = cv2.imread(mask_path)
        self.original_mask_copy = np.zeros(self.image.shape)
        self.original_mask_copy[np.where(self.original_mask != 0)] = 255
        self.mask = self.original_mask_copy.copy()
        self.to_move = False
        self.x0 = 0
        self.y0 = 0
        self.is_first = True
        self.xi = 0
        self.yi = 0
        self.window_name = "Move the mask. s:save; r:reset; q:quit"

    def _blend(self, image, mask):
        ret = image.copy()
        alpha = 0.3
        ret[mask != 0] = ret[mask != 0] * alpha + 255 * (1 - alpha)
        return ret.astype(np.uint8)

    def _move_mask_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.to_move = True
            if self.is_first:
                self.x0, self.y0 = x, y
                self.is_first = False
            self.xi, self.yi = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.to_move:
                M = np.float32([[1, 0, x - self.xi], [0, 1, y - self.yi]])
                self.mask = cv2.warpAffine(self.mask, M, (self.mask.shape[1], self.mask.shape[0]))
                cv2.imshow(self.window_name, self._blend(self.image, self.mask))
                self.xi, self.yi = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.to_move = False

    def move_mask(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._move_mask_handler)
        while True:
            cv2.imshow(self.window_name, self._blend(self.image, self.mask))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.image = self.image_copy.copy()
                self.mask = self.original_mask_copy.copy()
            elif key == ord("s"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()
        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)
        new_mask_path = path.join(path.dirname(self.image_path), 'target_mask.png')
        cv2.imwrite(new_mask_path, self.mask)
        cv2.destroyAllWindows()
        return self.xi - self.x0, self.yi - self.y0, new_mask_path

def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    mat_A.setdiag(-1, 1 * m)
    mat_A.setdiag(-1, -1 * m)
    return mat_A

def poisson_edit(source, target, mask, offset):
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0
    x_range = x_max - x_min
    y_range = y_max - y_min
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    source = cv2.warpAffine(source, M, (x_range, y_range))
    mask = mask[y_min:y_max, x_min:x_max]
    mask[mask != 0] = 1
    mat_A = laplacian_matrix(y_range, x_range)
    laplacian = mat_A.tocsc()
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()
    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()
        alpha = 1
        mat_b = laplacian.dot(source_flat) * alpha
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]
        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        target[y_min:y_max, x_min:x_max, channel] = x
    return target

def usage():
    print("Usage: python main.py [options] \n\n\
    Options: \n\
    \t-h\tPrint a brief help message and exits..\n\
    \t-s\t(Required) Specify a source image.\n\
    \t-t\t(Required) Specify a target image.\n\
    \t-m\t(Optional) Specify a mask image with the object in white and other part in black, ignore this option if you plan to draw it later.")

if __name__ == '__main__':
    print('main function starts running')
    args = {}
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hs:t:m:p:")
    except getopt.GetoptError as err:
        print(err)
        print("See help: main.py -h")
        exit(2)
    for o, a in opts:
        if o in ("-h"):
            usage()
            exit()
        elif o in ("-s"):
            args["source"] = a
        elif o in ("-t"):
            args["target"] = a
        elif o in ("-m"):
            args["mask"] = a
        else:
            assert False, "unhandled option"

    if ("source" not in args) or ("target" not in args):
        usage()
        exit()

    source = cv2.imread(args["source"])
    if source is None:
        print(f"Failed to load source image: {args['source']}")
        exit()

    target = cv2.imread(args["target"])
    if target is None:
        print(f"Failed to load target image: {args['target']}")
        exit()

    if source.shape[0] > target.shape[0] or source.shape[1] > target.shape[1]:
        print('Source image cannot be larger than target image.')
        exit()

    mask_path = ""
    if "mask" not in args:
        print('Please highlight the object to disapparate.\n')
        mp = MaskPainter(args["source"])
        mask_path = mp.paint_mask()
        print(f"Mask saved to: {mask_path}")
    else:
        mask_path = args["mask"]

    print('Please move the object to desired location to apparate.\n')
    mm = MaskMover(args["target"], mask_path)
    offset_x, offset_y, new_mask_path = mm.move_mask()
    print(f"New mask saved to: {new_mask_path} with offset ({offset_x}, {offset_y})")

    mask = cv2.imread(new_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load mask image: {new_mask_path}")
        exit()

    result = poisson_edit(source, target, mask, (offset_x, offset_y))
    output_path = path.join(path.dirname(args["target"]), 'result.png')
    cv2.imwrite(output_path, result)
    print(f"Result saved to: {output_path}")
