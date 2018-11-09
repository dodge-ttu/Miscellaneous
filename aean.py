import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

# /home/will/drone-code/2018-05-30_75_75_20_mauricio_odm_aoms_extracted.tif
# /home/will/drone-code/mauricio_field_map_ID_only_UPDATED.csv

def centers_contours(img, img_tag, thresh_value, area_filter):
    """ Apply a basic Open CV contouring alogrithm to find the contours and contour centers in a given
    image. This could be the prepared AOM sets or to count seedlings or flowers in a masked image.

    :param img: An image prepared in a way that is suitable to be contoured.
    :param img_tag: The image ID
    :return: The contour bounds, the contour centers, counts of contours and centers, the image tag.
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    (T, thresh) = cv2.threshold(img_gray, 0, thresh_value, 0)
    _, contours_raw, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [i for i in contours_raw if cv2.contourArea(i) > area_filter]
    contour_centers = []

    for idx, c in enumerate(contours):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        samp_bounds = cv2.boundingRect(c)
        contour_centers.append(((cX, cY), samp_bounds))

    contour_centers = sorted(contour_centers, key=lambda x: x[0])

    n_conts = len(contour_centers)
    n_cents = len(contours)

    print("[INFO] {0} contours and {1} contour centers found for image: {2}".format(n_conts, n_cents, img_tag))

    return contours, contour_centers, (n_conts, n_cents), img_tag


def green_diff_mask(img, img_tag):
    """ Returns an image mask and pixel count for pixels where the green value is greater than the blue
    and red (B < G > R)

    :param img: An image with vegetation intended to be analyzed.
    :param img_tag: the image ID
    :return: Masked copy, green pixel count data, image ID.
    """

    img_copy = img.copy()
    h, w, c = img_copy.shape
    b, g, r = cv2.split(img_copy)

    its_green = np.where(np.logical_and(g > b, g > r))

    x, y = its_green
    x = x.tolist()
    y = y.tolist()
    marks = [(x, y) for (x, y) in zip(x, y)]

    mask = np.zeros([h,w], dtype=np.uint8)

    for i in marks:
        mask[i] = 1

    img_copy = cv2.bitwise_and(img, img, mask=mask)

    num_grn_pix = len(marks)
    num_smp_pix = h * w
    percent_grn = round(num_grn_pix / num_smp_pix, 3)

    print("[INFO] {0} green pixels of {1} total pixels in sample: {2}".format(num_grn_pix, num_smp_pix, img_tag))

    return (img_copy, (marks), img_tag)



class CropAnalyzer:
    """ Crop analyzer class with methods to extract, tag, and analyze sample spaces in prepared
    drone imagery. This class is specific to composited drone imagery prepared by the method described
    in the documentation associated with the crop analyzer project. """

    def __init__(self, aom_set_path, num_aom_rows, num_aom_tiers, germ_id_set_path, output_folder_path, analysis_id):

        self.analysis_id = analysis_id
        self.aom_set_path = aom_set_path
        self.num_aom_rows = num_aom_rows
        self.num_aom_tiers = num_aom_tiers
        self.germ_id_set_path = germ_id_set_path
        self.output_folder_path = output_folder_path
        self.image_out_format = '.png'
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.germ_id_set = None
        self.aom_set = None
        self.aom_set_marked = None
        self.organized_centers_contours = None
        self.aom_row_tier_tags = None
        self.green_diff_mask_sets = None
        self.plant_count_data = None
        self.samples_dir = None
        self.green_diff_dir = None
        self.plants_counted_dir = None

        print("[INFO] CropAnalyzer instance created for aom set with the following path: {0}".format(self.aom_set_path))

    def germ_id_modify(self):
        df = pd.read_csv(self.germ_id_set_path, header=[0])
        df_long = pd.melt(df)
        df_long.dropna(inplace=True)

        self.germ_id_set = [i for i in df_long["value"].values]

        print("[INFO] Germplasm ID set length is: {0}".format(len(self.germ_id_set)))

    def read_aom_set(self):
        self.aom_set = cv2.imread(self.aom_set_path)

        h, w, c = self.aom_set.shape

        print("[INFO] Image with shape == height: {0} width: {1} channels: {2} has been read".format(h,w,c))

    def get_set_contours_centers(self, thresh_val, area_filt):

        contours, contour_centers, (n_conts, n_cents), img_tag = centers_contours(self.aom_set, self.analysis_id, thresh_value=thresh_val, area_filter=area_filt)

        contours_ls = [contours, contour_centers, (n_conts, n_cents), img_tag]

        # after this:
        # self.contours_ls[0] = contours
        # self.contours_ls[1] = contour_centers
        # self.contours_ls[2][0] = n_conts
        # self.contours_ls[2][1] = n_cents
        # self.contours_ls[3] = img_tag

        self.contours_ls = contours_ls

    def group_contours(self):
        rows = self.num_aom_rows

        self.grouping_sequence = [i for i in range(rows, len(self.contours_ls[1])+(rows+1), rows)]

        print("[INFO] Length of grouping sequence: {0}".format(len(self.grouping_sequence)))
        print("[INFO] First number of each group: {0}".format(self.grouping_sequence))

    def organize_centers(self):
        organized_centers_contours = []
        for idx, i in enumerate(self.grouping_sequence):
            print("[DEBUG] Organizing contour centers for tier {0}".format(idx))
            if idx == 0:
                row = self.contours_ls[1][:i]
                row = sorted(row, key=lambda x: x[0][1])
                organized_centers_contours.extend(row)
            else:
                row = self.contours_ls[1][self.grouping_sequence[idx - 1]:i]
                row = sorted(row, key=lambda x: x[0][1])
                organized_centers_contours.extend(row)

        self.organized_centers_contours = organized_centers_contours

        print("[INFO] Length of organized contours and centers list: {0}".format(len(self.organized_centers_contours)))

    def make_row_tier_tags(self):
        row_tier_tuple_list = []

        for i in range(1, self.num_aom_tiers + 1, 1):
            if i == 56:
                for j in range(1, ((self.num_aom_rows + 1) - 2), 1):
                    row_tier_tuple_list.append("row-{0}__tier-{1}".format(j, i))
            else:
                for j in range(1, self.num_aom_rows + 1, 1):
                    row_tier_tuple_list.append("row-{0}__tier-{1}".format(j, i))

        self.aom_row_tier_tags = row_tier_tuple_list

        print("[INFO] Row tier tag length: {0}".format(len(self.aom_row_tier_tags)))

    def put_id_on_sample_map(self):
        self.aom_set_marked = self.aom_set.copy()
        for center, id, row_tier in zip(self.organized_centers_contours, self.germ_id_set, self.aom_row_tier_tags):
            cv2.putText(self.aom_set_marked, str(id), (center[0][0] + 120, center[0][1] + 40), self.font, .6, (200, 0, 0), 1)
            cv2.putText(self.aom_set_marked, str(row_tier), (center[0][0] - 240, center[0][1] + 40), self.font, .6, (200, 0, 0), 1)

        print("[INFO] Sample map text tags inserted.")

    def make_sample_images(self):
        """
        :param img_input: masked composite ready for processing
        :return: list of image slices derived from cv2.boundingRect()
        """

        smple_images = []

        for i, id_tag, row_tier_tag in zip(self.organized_centers_contours, self.germ_id_set, self.aom_row_tier_tags):
            x = i[1][0]
            y = i[1][1]
            w = i[1][2]
            h = i[1][3]

            img = self.aom_set[y:(y + h), x:(x + w)]
            # cv2.rectangle(self.aom_set, (x, y), (x + w, y + h), (0, 255, 0), 2)
            smple_images.append((img, (id_tag, row_tier_tag)))

        self.sample_images = smple_images

        print("[INFO] {0} sample images extracted from aom set: {1}".format(len(smple_images), self.analysis_id))

    def make_green_diff_set(self):
        """ A method to apply the green diff mask function to the list of sample spaces extracted from
        the AOM set."""

        gd_mask_imgs = []

        for (img, (id_tag, row_tier_tag)) in self.sample_images:
            (gdimg, (marks), tag) = green_diff_mask(img, (id_tag, row_tier_tag))
            gd_mask_imgs.append((gdimg, (marks), (id_tag, row_tier_tag)))

        self.green_diff_mask_sets = gd_mask_imgs

        print("[INFO] {0} green differential masks extracted from set: {1}".format(len(gd_mask_imgs), self.analysis_id))


    def count_plants(self, thresh_val, area_filt):
        """A method to count plants from prepared composite imagery. The intent is for this method to be used on
        early season imagery.
        """

        counted_plants_samples = []

        for (img, (marks), (id_tag, row_tier_tag)) in self.green_diff_mask_sets:
            pc_conts, pc_cents, (pn_conts, pn_cents), tag = centers_contours(img, (id_tag, row_tier_tag), thresh_value=thresh_val, area_filter=area_filt )
            img = cv2.drawContours(img.copy(), pc_conts, -1, (0,255,0), 1)
            counted_plants_samples.append((img, (pc_conts, pc_cents, (pn_conts, pn_cents), (id_tag, row_tier_tag), (marks))))
            print("[INFO] {0} plants counted for germplasm at: {1}".format(pn_conts, (id_tag, row_tier_tag)))

        self.plant_counts_and_marked_img = counted_plants_samples

    def make_plant_count_df(self):
        """ Make a pandas dataframe out of the plant count data generated from self.count_plants method.
        """

        x_vals = []
        y_vals = []
        id_tags = []
        row_tier_tags = []
        rt_and_ids = []
        areas_ls = []
        area_sums =  []

        for (img, (pc_conts, pc_cents, (pn_conts, pn_cents), (id_tag, row_tier_tag), (marks))) in self.plant_counts_and_marked_img:

            x = [i[0][0] for i in pc_cents]
            y = [i[0][1] for i in pc_cents]
            x_green = [i[0] for i in marks]
            y_green = [i[1] for i in marks]

            n = len(x)

            area = [cv2.contourArea(i) for i in pc_conts]
            area_sum = [sum(area)] * n
            ids = [id_tag] * n
            row_tiers = [row_tier_tag] * n
            rt_and_id = ['{0}__{1}'.format(id_tag, row_tier_tag)] * n

            x_vals.extend(x)
            y_vals.extend(y)
            id_tags.extend(ids)
            row_tier_tags.extend(row_tiers)
            rt_and_ids.extend(rt_and_id)
            areas_ls.extend(area)
            area_sums.extend(area_sum)


        dict = {
            'germ_ID': id_tags,
            'row_tier_tags': row_tier_tags,
            'id_and_rt': rt_and_ids,
            'x':x_vals,
            'y':y_vals,
            'area':areas_ls,
            'area_sum':area_sums,

        }

        df_contour = pd.DataFrame(dict)

        x_green_vals = []
        y_green_vals = []
        id_tags = []
        row_tier_tags = []
        rt_and_ids = []

        for (img, (pc_conts, pc_cents, (pn_conts, pn_cents), (id_tag, row_tier_tag), (marks))) in self.plant_counts_and_marked_img:

            x_green = [i[1] for i in marks]
            y_green = [i[0] for i in marks]

            n = len(x_green)

            ids = [id_tag] * n
            row_tiers = [row_tier_tag] * n
            rt_and_id = ['{0}__{1}'.format(id_tag, row_tier_tag)] * n

            x_green_vals.extend(x_green)
            y_green_vals.extend(y_green)
            id_tags.extend(ids)
            row_tier_tags.extend(row_tiers)
            rt_and_ids.extend(rt_and_id)

        dict = {
            'germ_ID': id_tags,
            'row_tier_tags': row_tier_tags,
            'id_and_rt': rt_and_ids,
            'x_green': x_green_vals,
            'y_green': y_green_vals,
        }

        df_green_pix = pd.DataFrame(dict)

        self.plant_count_data = (df_contour, df_green_pix)


    def write_sample_map(self):
        cv2.imwrite(os.path.join(self.output_folder_path, self.analysis_id + self.image_out_format), self.aom_set_marked)

    def write_samples(self):

        directory = os.path.join(self.output_folder_path, self.samples_dir)

        if not os.path.exists(directory):
            os.makedirs(directory)

        for (img, (id_tag, row_tier_tag)) in self.sample_images:
            cv2.imwrite(os.path.join(directory,'{0}_{1}_COUNTED{2}'.format(id_tag,row_tier_tag,self.image_out_format)), img)


    def write_green_diff_samples(self):

        directory = os.path.join(self.output_folder_path, self.green_diff_dir)

        if not os.path.exists(directory):
            os.makedirs(directory)

        for (gdimg, (marks), (id_tag, row_tier_tag)) in self.green_diff_mask_sets:
            cv2.imwrite(os.path.join(directory,'{0}_{1}_COUNTED{2}'.format(id_tag,row_tier_tag,self.image_out_format)), gdimg)

    def write_counted_plant_samples(self):

        directory = os.path.join(self.output_folder_path, self.plants_counted_dir)

        if not os.path.exists(directory):
            os.makedirs(directory)

        for (img, (pc_conts, pc_cents, (pn_conts, pn_cents), (id_tag, row_tier_tag), (marks))) in self.plant_counts_and_marked_img:
            cv2.imwrite(os.path.join(directory,'{0}_{1}_COUNTED{2}'.format(id_tag,row_tier_tag,self.image_out_format)), img)















    


