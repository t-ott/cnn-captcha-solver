import os
import cv2
import matplotlib.pyplot as plt

class Segmenter:
    '''
    Methods
    -------
    segment_chars(img_path, plot=False):
        Utilizes OpenCV findContours() to extract bounding rectangles of
        characters within a CAPTCHA image

    plot_segmented_chars(img, segmented_chars):
        Plot image and segmented characters side-by-side
    '''
    def segment_chars(self, img_path: str, plot=False) -> list:
        '''
        Utilizes OpenCV findContours() to extract bounding rectangles of
        characters within a CAPTCHA image

        Parameters
        ----------
        img_path : str
            Path to CAPTCHA image to segment into characters
        plot : bool, default: False
            Option to plot the CAPTCHA segmentation results

        Returns
        -------
        segmented_chars : list
            List of (char_img, label) tuples
        '''
        img = cv2.imread(img_path)
        # covert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # take binary threshold
        ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        # invert image
        bit_not = cv2.bitwise_not(thresh)
        # find contours
        contours, hierarchy = cv2.findContours(bit_not, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # get bounding rect of each contour
        rects = [cv2.boundingRect(c) for c in contours]
        # sort rects by their width
        rects.sort(key=lambda x: x[2])

        # deal with touching letters where one wide bounding box
        # envlopes two letters. split these in half
        while len(rects) < 4:
            # pop widest rect
            wide_rect = rects.pop()
            x, y, w, h = wide_rect
            # split in two
            first_half = (x, y, w//2, h)
            second_half = (x+w//2, y, w//2, h)
            rects.append(first_half)
            rects.append(second_half)
            # re-sort by width
            rects.sort(key=lambda x: x[2])

        if len(rects) > 4:
            print('For some reason more than 4 characters were identified in '
                  'the CAPTCHA! Returning the widest four...')
            rects = rects[-4:]

        # sort rects by horizontal position left to right
        rects.sort(key=lambda x: x[0])
        # define labels using each char from filename string
        labels = [char for char in os.path.basename(img_path).split('.')[0]]

        # use bounding rects to crop each character from the image array
        segmented_chars = []
        for rect, label in zip(rects, labels):
            x, y, w, h = rect

            # buffer char's bounding rect by one pixel if possible
            if x > 0:
                x -= 1
            if y > 0:
                y -= 1
            if x+w < img.shape[1]:
                w += 1
            if y+h < img.shape[0]:
                h += 1

            char_img = img[y:y+h, x:x+w, :]
            segmented_chars.append((char_img, label))

        if plot:
            self.plot_segmented_chars(img, segmented_chars)

        return segmented_chars


    def plot_segmented_chars(self, img, segmented_chars):
        '''
        Plot image and segmented characters side-by-side.

        Parameters
        ----------
        img : np.array
            CATPCHA image array
        segmented_chars: list
            List of (char_img, label) tuples
        '''

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            1, 5, figsize=(12,2), gridspec_kw={'width_ratios': [4, 1, 1, 1, 1]}
        )
        ax1.imshow(img)
        char_imgs = [segmented_char[0] for segmented_char in segmented_chars]
        ax2.imshow(char_imgs[0])
        ax3.imshow(char_imgs[1])
        ax4.imshow(char_imgs[2])
        ax5.imshow(char_imgs[3])
