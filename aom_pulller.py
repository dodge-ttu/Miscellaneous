print("="*100)
print("="*100)
print("""
Will Dodge | 2018 | dodge.ttu@gmail.com
""")
print("="*100)
print("="*100)
print("begining processing....")

print("importing libraries...")
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import pathlib
import os
from os.path import isfile, join

# input path
aom_extracted_path = "C:\\Users\\dodge\\Desktop\\AOMs extracted\\AOM extracted sets\\"

# output path
path = "C:\\Users\\dodge\\Desktop\\AOMs extracted\\"

filenames = os.listdir(aom_extracted_path)

print("reading in large aom extracted sets...")
images = [cv2.imread(aom_extracted_path + i) for i in filenames]

print("reading germplasm ID data")
germplasm_id_set_path = "C:\\Users\\dodge\\Desktop\\AOMs extracted\\mauricio_field_map_ID_only_UPDATED.csv"

df = pd.read_csv(germplasm_id_set_path, header=[0])

df_long = pd.melt(df)

df_long.dropna(inplace=True)

print("germplasm ID set length is {0}".format(len(df_long)))

germplasm_id_set = [i for i in df_long["value"].values]

font = cv2.FONT_HERSHEY_DUPLEX

def find_contours_and_centers(img_input):

    """
    :param img_input: composite with AOMs extracted ready for analyisis
    :return: a list of contours and list of contour center tuples

    """
    print("find contours and centers starting...")
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
    (T, thresh) = cv2.threshold(img_gray, 0, 255, 0)
    _, contours_raw, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [i for i in contours_raw if cv2.contourArea(i) > 50000]
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

contours_all = []
centers_and_bounds_all  = []

for image_set in images:
    contours, centers = find_contours_and_centers(image_set)
    contours_all.append(contours)
    centers_and_bounds_all.append(centers)

grouping_sequence_all = []

for idx, centers in enumerate(centers_and_bounds_all):
    print("generating grouping sequence {0}".format(idx))
    grouping_sequence = [i for i in range(56, len(centers)+57, 56)]
    grouping_sequence_all.append(grouping_sequence)

print("grouping_sequence length {0}".format(len(grouping_sequence_all)))

organized_contour_centers_and_bounds_all = []

for idx1, (grouping_sequence, centers_and_bounds) in enumerate(zip(grouping_sequence_all, centers_and_bounds_all)):
    print("organizing contour centers for image set {0}".format(idx1))
    print("=" * 100)
    print("=" * 100)
    organized_contour_centers_and_bounds = []
    for idx, i in enumerate(grouping_sequence):
        print("organizing contour centers for tier {0}".format(idx))
        if idx == 0:
            row = centers_and_bounds[:i]
            row = sorted(row, key=lambda x: x[0][1])
            organized_contour_centers_and_bounds.extend(row)
        else:
            row = centers_and_bounds[grouping_sequence[idx-1]:i]
            row = sorted(row, key=lambda x: x[0][1])
            organized_contour_centers_and_bounds.extend(row)

    organized_contour_centers_and_bounds_all.append(organized_contour_centers_and_bounds)

print("="*100)
print("organized contour center length {0}".format(len(organized_contour_centers_and_bounds_all)))
print("="*100)


def make_row_tier_tags(number_of_tiers, number_of_rows):

    """
    :param number_of_tiers: number of tiers in map
    :param number_of_rows: number of rows in map
    :return: list of of IDs that are organized for the purpose of "pasting" on our images.
    """

    row_tier_tuple_list = []

    for i in range(1, number_of_tiers + 1, 1):
        if i == 56:
            for j in range(1, ((number_of_rows + 1) - 2), 1):
                row_tier_tuple_list.append("row-{0}__tier-{1}".format(j, i))
        else:
            for j in range(1, number_of_rows + 1, 1):
                row_tier_tuple_list.append("row-{0}__tier-{1}".format(j, i))

    return row_tier_tuple_list


row_tier_tags = make_row_tier_tags(25, 56)

# Delete tags from tag set for plots that are missing or do not come up

del row_tier_tags[1344:1345]

def put_id_on_sample_map(img_input, contour_centers_and_bounds, id_set, row_tier_list):


    """
    :param img_input:
    :param contour_centers_and_bounds:
    :return: Nothing. The main composite is marked.

    """

    for center, id, row_tier in zip(contour_centers_and_bounds, id_set, row_tier_list):
        cv2.putText(img_input, str(id), (center[0][0]+120, center[0][1]+40), font, .6, (200, 0, 0), 1)
        cv2.putText(img_input, str(row_tier), (center[0][0] - 240, center[0][1] + 40), font, .6, (200, 0, 0), 1)


for idx, (org_centers_and_bounds, i) in enumerate(zip(organized_contour_centers_and_bounds_all, images)):
    print("tagging image set {0}".format(idx))
    put_id_on_sample_map(i, org_centers_and_bounds, germplasm_id_set, row_tier_tags)

def make_sample_images(img_input, contour_bounds):

    """
    :param img_input: masked composite ready for processing
    :return: list of image slices derived from cv2.boundingRect()
    """

    smple_images = []

    for i in contour_bounds:
        x = i[1][0]
        y = i[1][1]
        w = i[1][2]
        h = i[1][3]

        img = img_input[y:(y + h), x:(x + w)]
        #cv2.rectangle(img_input, (x, y), (x + w, y + h), (0, 255, 0), 2)
        smple_images.append(img)

    return smple_images


sample_image_sets_all = []

for idx, (i,j) in enumerate(zip(images, organized_contour_centers_and_bounds_all)):
    print("generating image sample set {0}".format(idx))
    sample_images = make_sample_images(i, j)
    sample_image_sets_all.append(sample_images)

print("generating image stacks for sample sets")

min_set_length = min([len(i) for i in sample_image_sets_all])

print("minimum set length is {0}".format(min_set_length))

unbound_sample_image_stacks_all = []

for i in range(min_set_length - 1):
    stack_set = []
    for j in sample_image_sets_all:
        img = cv2.resize(j[i], (600, 100))
        stack_set.append(img)

    unbound_sample_image_stacks_all.append(stack_set)

bound_stack_sets = []

for i in unbound_sample_image_stacks_all:
    img_stacks = np.vstack(i)
    bound_stack_sets.append(img_stacks)

for idx, (name, sample_image_set, aom_extracted_image) in enumerate(zip(filenames, sample_image_sets_all, images)):
    directory = path + name + "_folder"
    print("writing image set {0}".format(idx))
    if not os.path.exists(directory):
        os.makedirs(directory)
        for (germplasm_id, img, row_tier) in zip(germplasm_id_set, sample_image_set, row_tier_tags):
            cv2.imwrite(os.path.join(directory, (germplasm_id + "_" + row_tier + ".tif")), img)

directory_maps = os.path.join(path, "aom_maps\\")
if not os.path.exists(directory_maps):
    os.makedirs(directory_maps)

for (name, aom_extracted_image) in zip(filenames, images):
    print("writing map {0}".format(name))
    cv2.imwrite(os.path.join(directory_maps, (name + "_map.tif")), aom_extracted_image)

directory_for_stacks_all = path + "2018_sample_image_stacks\\"
print("making stack image set directory...")
if not os.path.exists(directory_for_stacks_all):
    os.makedirs(directory_for_stacks_all)

print("writing stack image sets...")

for germplasm_id, row_tier, i in zip(germplasm_id_set, row_tier_tags, bound_stack_sets):
    cv2.imwrite(directory_for_stacks_all + germplasm_id + "_" + row_tier + "_" + "_stack.png", i)

print("="*50)

print("="*50)

print("clearing objects from environment to free up memory...")

del centers_and_bounds_all, contours_all, images, organized_contour_centers_and_bounds_all

print("="*50)
print("="*50)

print("generating growth data sets for each variety...")


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


def plant_count(green_diff_morph):
    kernel = np.ones((5, 5), np.uint8)
    green_diff_morph_gray = cv2.cvtColor(green_diff_morph, cv2.COLOR_BGR2GRAY)
    (T, thresh) = cv2.threshold(green_diff_morph, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (5,5))
    (_, plants, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    return green_diff_morph


sample_image_set_directories = [path + name + "_folder\\" for name in filenames]

sample_image_filenames_all = [os.listdir(directory_path) for directory_path in sample_image_set_directories]

sample_image_filepaths_all = []

for directory_path, samp_filenames in zip(sample_image_set_directories, sample_image_filenames_all):
    samp_filepaths = [os.path.join(directory_path, i) for i in samp_filenames]
    sample_image_filepaths_all.append(samp_filepaths)

sample_images_all = []

for idx, i in enumerate(sample_image_filepaths_all):
    print("reading in sample image set {0}".format(idx))
    img_set = [cv2.imread(j) for j in i]
    sample_images_all.append(img_set)

green_diff_mask_sets_all = []

for idx, i in enumerate(sample_images_all):
    print("applying green diff mask in sample image set {0}".format(idx))
    green_diff_sets = [green_diff_mask(j) for j in i]
    green_diff_mask_sets_all.append(green_diff_sets)

for idx, (name, sample_image_set) in enumerate(zip(filenames, green_diff_mask_sets_all)):
    directory = path + "GREEN_DIFF_" + name + "_folder\\"
    print("writing image set {0}".format(idx))
    if not os.path.exists(directory):
        os.makedirs(directory)
        for (germplasm_id, img, row_tier) in zip(germplasm_id_set, sample_image_set, row_tier_tags):
            cv2.imwrite(os.path.join(directory, (germplasm_id + "_" + row_tier + ".tif")), img[0])


flight_date_list = [i[:10] for i in filenames]

dict = {"flight_date": pd.to_datetime(flight_date_list)}

df_growth_data = pd.DataFrame(dict)

unbound_green_diff_pixel_data_all = []

for i in range(min_set_length - 1):
    stack_set = []
    for j in green_diff_mask_sets_all:
        h, w, c = j[i][0].shape
        green_diff_pixel_count = j[i][1]
        total_sample_pixel_count = h * w
        stack_set.append((green_diff_pixel_count, total_sample_pixel_count))

    unbound_green_diff_pixel_data_all.append(stack_set)

for i, j in zip(germplasm_id_set, unbound_green_diff_pixel_data_all):
        green_diff_pix_count = [k[0] for k in j]
        total_smple_pixel_count = [k[1] for k in j]
        percent_canopy_cover = [a/b for a,b in zip(green_diff_pix_count, total_smple_pixel_count)]
        df_growth_data[i] = percent_canopy_cover

df_growth_data.set_index("flight_date", inplace=True)

plt.figure(figsize=(24,18))
for i in list(df_growth_data):
    plt.plot(df_growth_data[i])

plt.savefig(os.path.join(path, "all_germ_growth_plot.png"))

plot_sets = df_growth_data[list(df_growth_data)[:600]]

fig, axes = plt.subplots(20, 30, figsize=(24, 18), sharey=True, sharex=True)
fig.subplots_adjust(hspace=.3, wspace=.175)
for ax, vector_name in zip(axes.ravel(), list(plot_sets)):
    ax.plot(plot_sets[vector_name], "r-")
    #ax.set_title(vector_name, {"fontsize":3})
    ax.tick_params(axis='x', labelrotation=90, labelsize=2)
    ax.axis("off")

fig.savefig(os.path.join(path, "small_multiple_germ_growth_plot.png"))

print("TT Guns Up TT")

