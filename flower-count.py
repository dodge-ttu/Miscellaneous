import cv2
import numpy as np


img = cv2.imread("/home/will/plant-count-test.tif")
#img = cv2.imread("/home/will/flower-count-test.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
masked = cv2.bitwise_and(img, img, mask=thresh)

def find_contours_and_centers(img_input):

    """
    :param img_input: composite with AOMs extracted ready for analyisis
    :return: a list of contours and list of contour center tuples

    """
    print("find contours and centers starting...")
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
    (T, thresh) = cv2.threshold(img_gray, 0, 100, 0)
    _, contours_raw, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [i for i in contours_raw if cv2.contourArea(i) > 2]
    contour_centers = []

    for idx, c in enumerate(contours):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        samp_bounds = cv2.boundingRect(c)
        contour_centers.append(((cX,cY), samp_bounds))

    print("{0} contour centers and bounds found".format(len(contour_centers)))

    contour_centers = sorted(contour_centers, key=lambda x: x[0])
    print("...done")

    return contours, contour_centers


def green_diff_mask(sample_image):

    """
    Returns an image mask and pixel count for pixels where the green value is greater than the blue and red (B < G > R)

    :param sample_image:
    :return: masked image and pixel count
    """

    img_original = sample_image.copy()
    b, g, r = cv2.split(sample_image)
    # mask = np.where(np.logical_and((img[:,:,1] > img[:,:,0]), (img[:,:,1] > img[:,:,2])))

    # b = b.astype(np.int16)
    # g = g.astype(np.int16)
    # r = r.astype(np.int16)

    mask = np.where(np.logical_and(g > b, g > r))

    x, y = mask
    x = x.tolist()
    y = y.tolist()
    marks = [(x, y) for (x, y) in zip(x, y)]

    img_marked = sample_image.copy()

    for i in marks:
        # cv2.circle(img, (i[1], i[0]), 1, (255,255,255), 1)
        img_marked[i] = (255, 255, 255)

    img_marked = cv2.cvtColor(img_marked, cv2.COLOR_BGR2GRAY)
    (T, mask) = cv2.threshold(img_marked, 254, 255, cv2.THRESH_BINARY)

    img_out = cv2.bitwise_and(img_original, img_original, mask=mask)

    return img_out, len(marks)

img_gdmask = green_diff_mask(img)

conts, centers = find_contours_and_centers(img_gdmask[0])

marked = cv2.drawContours(img_gdmask[0], conts, -1, (0, 255, 0), 1)

font = cv2.FONT_HERSHEY_SIMPLEX

for idx, i in enumerate(centers):
    cv2.putText(marked, str(idx), (i[0][0]-20,i[0][1]+20), font, .4, (0, 255, 0), 2)

cv2.imwrite('/home/will/plant-count-out2.png', marked)
