# This code has been authored by myself, Will Dodge. Started 2017 building some functions to analyze drone image data.

#/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import argparse
import numpy as np
import math
import os
from os import listdir
from os.path import isfile, join
import glob

def get_loc(fld_img, mrk_tup_L, mrk_tup_H):
    """ Return the locations of all pixels with color tuples that fall with fall within the given range

    :param fld_img: composite field image
    :param mrk_tup_L: tuple representing low end of marker color range
    :param mrk_tup_H: tuple representing high end of marker color range
    :return: list of ordered pairs for each pixel that falls within given range
    """

    loc = np.where(np.all(np.logical_and((fld_img >= mrk_tup_L), (fld_img <= mrk_tup_H)), axis=-1))
    x, y = loc
    x = x.tolist()
    y = y.tolist()
    marks = [(x, y) for (x, y) in zip(x, y)]

    return marks

def get_one_xy_per_mark(fld_img, mark_locs):
    """ Return a single ordered pair representing the center of each marker. This is done my using the open
    cv contour detection feature to contour each marker and return the center point from the list of
    statistics from each contour that cv2.findContours() returns.

    :param fld_img: composite field image
    :param mark_locs: list of all pixel coordinates that relate to the pixels that fall in the given range
    :return list containing single ordered pair for each marker


    """
    def enhance_marker(fld_img, mark_locs, enhance_radius, color_tuple):
        """ Draw a circle with given radius at the given pixel locations. For the purpose of making markers white so
        they are easily contoured and marker center can be extracted, ultimately becoming anchor point for sample space.


        :param fld_img: composite field image
        :param marks_1: list of coordinates for all the pixels with BGR values in given range from get_loc()
        :param enhance_radius: radius of circle to be placed at marker pixels
        :param color_tuple: RGB color tuple given in form of (B,G,R)
       """
        for i in mark_locs:
            cv2.circle(fld_img, (i[1], i[0]), enhance_radius, color_tuple, -1)

    (h, w, c) = fld_img.shape
    spots = np.zeros((h, w, 3), dtype = "uint8")
    enhance_marker(spots, mark_locs, 8, (255, 255, 255))
    spots = cv2.cvtColor(spots,cv2.COLOR_BGR2GRAY)
    (_, cnts, _) = cv2.findContours(spots.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks_moments = [cv2.moments(i) for i in cnts]
    marks_clean = [(int(M["m01"] / M["m00"]), int(M["m10"] / M["m00"])) for M in marks_moments]

    return marks_clean

def draw_samp_rectangle(fld_img, marks_1, marker_height, marker_width):
    h, w = (marker_height, marker_width)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, i in enumerate(marks_1):
        cv2.rectangle(fld_img, (i[1] + 3*w, i[0] + round(.5*h)), (i[1] - 3*w, i[0] - 8*h), (0, 255, 0), 8)
        cv2.putText(fld_img, str(idx), (i[1], i[0] + 240), font, 10, (230, 230, 0), 28)
        cv2.putText(fld_img, str(idx), (i[1] + round(.70*w), i[0] - 10), font, 1.5, (230, 230, 0), 4)
        cv2.putText(fld_img, 'dodge.ttu@gmail.com', (200, 200), font, 2, (0, 242, 255), 3)

    return fld_img

def draw_samp_circles(fld_img, marks_1, samp_rad):
    """ Return a field image with circles drawn with centers located at ordered pairs given in the form of list. These
    circles represent the bounds of the sample sites. There are numbered in the order in which they are passed to the
    function

    :param fld_img: composite field image
    :param marks_1: list of ordered pairs from which sample space is extracted
    :param samp_rad: integer for sample circle radius
    :return: field image with circles drawn and numbered
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, i in enumerate(marks_1):
        cv2.circle(fld_img, (i[1],i[0]), samp_rad, (0, 255, 0), 10)
        cv2.putText(fld_img, str(idx), (i[1] + 200, i[0] - 10), font, 10, (0, 242, 255), 20)
        cv2.putText(fld_img, str(idx), (i[1]-54, i[0]+(samp_rad - 20)), font, 3, (0, 242, 255), 5)
        cv2.putText(fld_img, 'dodge.ttu@gmail.com', (200,200), font, 2, (0, 242, 255), 3)
    return fld_img
def get_circle_samps(fld_img, marks_1, samp_rad):
    """ Crop out sample image spaces as a square and mask so that only the circular sample space is visible

    :param fld_img: composite field image
    :param marks_1: pixel location of sample sites in the form of a list of ordered pairs (x,y)
    :param samp_rad: sample circle radius
    :return: list of ordered pairs that represent the center of each marker
    """

    mask = np.zeros((2*samp_rad, 2*samp_rad), dtype=np.uint8)
    cv2.circle(mask, (samp_rad, samp_rad), samp_rad, 255, -1)
    samp_bounds = [((i[0]-samp_rad, i[0]+samp_rad), (i[1]-samp_rad, i[1]+samp_rad)) for i in marks_1]
    fin_samp = [fld_img[i[0][0]:i[0][1], i[1][0]:i[1][1]] for i in samp_bounds]
    fin_samp = [cv2.bitwise_and(i, i, mask = mask) for i in fin_samp]

    return fin_samp
def get_rectangle_samps(fld_img, marks_1, marker_height, marker_width):
    """ Return a list of ordered pairs for sample spaces derived from marker coordinates. These sample spaces are
    generated as a multiple of the marker size in each image

    :param marks_1:
    :return:
    """
    h = marker_height
    w = marker_width

    samp_bounds = [((i[0] - 8*h, i[0] + round(.5*h)), (i[1] - 3*w, i[1] + 3*w)) for i in marks_1]
    fin_samp = [fld_img[i[0][0]:i[0][1], i[1][0]:i[1][1]] for i in samp_bounds]

    return fin_samp
def write_samps(fin_samp, img_mrked, fld_img_ID, out_folder_path):
    """ Write out sample images in jpg form from list of images stored as numpy array

    :param fin_samp: list of individual sample site images
    :param img_marked: marked image returned from draw_samp_cirlces()
    :param out_folder_path: directory to output sample site image set
    :param fld_img_ID: field image ID to insert into sample thumbnail and output file names
    """

    for idx, i in enumerate(fin_samp):
        cv2.imwrite(out_folder_path + '\\' + fld_img_ID + '_sample_' + str(idx) + '.jpg', i)
        print('sample image {0} write complete'.format(idx))
    print('writing sample site map...')
    cv2.imwrite(output_folder_path + '\\' + 'WGD_fld_img_MAP' + composite_id + '.jpg', img_mrked)
    print('sample map write complete! Wreck Em!')

def estimate_yield_for_sample_spaces(in_folder_path, cutoff_val, out_folder_path):
    print('Reading sample images...')
    samp_filename = [f for f in listdir(in_folder_path) if isfile(join(in_folder_path, f))]
    sample_image_list = [cv2.imread(in_folder_path + '/' + i) for i in samp_filename]
    print('Estimating yield and writing processed samples......')
    for idx, i in enumerate(sample_image_list):
        img = i
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gb = cv2.GaussianBlur(img_g, (5, 5), 0)
        (T, cutoff) = cv2.threshold(img_gb, cutoff_val, 255, cv2.THRESH_BINARY)
        # (T, cutoff) = cv2.threshold(img_gb, 180, 255, cv2.THRESH_BINARY)
        loc_2 = np.where(255 == cutoff)
        for pt in zip(*loc_2[::-1]):
            img[pt[1], pt[0]] = (0, 0, 255)
        a, b = loc_2
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(len(a)) + " pixels", (8, 26), font, .75, (0,242,255), 2, cv2.LINE_AA)
        cv2.putText(img, str(round((-2E-06*(len(a))**2 + 0.0723*(len(a)) + 101.67), 3)) + " g/m2", (8, 66), font, .75, (0, 242, 255), 2, cv2.LINE_AA)
        # cv2.putText(img, str(samp_filename[idx]), (100, 26), font, .5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(out_folder_path + '/WGD_sample_counted_{0}_{1}.jpg'.format(samp_filename[idx],
                                                                               cutoff_val), img)
    print('File write complete!')


def make_binned_heat_map(fld_img, bin_size_height, bin_size_width, cutoff_val):
    """ Make heat map by, first: slicing the image into extremely small slices (20 by 20, etc.), second: performing
    yield estimate on each slice and storing that value in a list, third: assigning the section of the original image
    represented by each slice a solid color based on the which bin the pixel yield estimate.

    :param fld_img: composite field image
    :param bin_size_height: height to be used for image slices to be binned
    :param bin_size_width: width to be used for image slices to be banned
    :param cutoff_val: threshold value for pixel yield estimate
    :return: field image heat map with seven bins with distinct colors
    """

    def get_fld_img_sect_cords(fld_img, bin_size_height, bin_size_width):
        """ Take a composite field image and list of coordinates so that the field image can be divided into chunks of
        given height and width. For the purpose of 'binning' each piece and assigning a color.

         :param fld_img: composite field image
         :param bin_size_height: height in pixels of image sections to be 'binned'
         :param bin_size_width: width in pixels of image section to be 'binned'
         :return list of coordinates to slice composite field image for 'binning'
        """

        fld_img_h, fld_img_w, fld_img_c = fld_img.shape
        fld_img_h_s = range(0, fld_img_h, bin_size_height)
        fld_img_w_s = range(0, fld_img_w, bin_size_width)
        fld_img_sect_cords = []
        for y in fld_img_h_s:
            for x in fld_img_w_s:
                fld_img_sect_cords.append((x, y))

        return fld_img_sect_cords

    def get_fld_img_sections(fld_img, fld_img_sect_cords, bin_size_height, bin_size_width):
        """ Use the list of coordinates returned from get_fld_img_sect_cords() to slice image array into individual
        images with proportions equal to bin_size_height and bin_size_width stored in a list

        :param fld_img: composite field image
        :param fld_img_sect_cords: a list of image
        :param bin_size_height:
        :param bin_size_width:
        :return: list of coordinates with which image can be sliced for 'binning'
        """

        fld_img_sections = []
        for i in fld_img_sect_cords:
            fld_img_sections.append(fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width])

        return fld_img_sections

    def get_pix_count_for_img_sections(fld_img_sections, cutoff_val):
        """ Iterate over list of image slices, returned from get_fld_image_sections(), and calculate pixel count for
        that region of the image, then append that value to a list of pixel yield values, then return values.

        :param fld_img_sections: list of sliced field image sections
        :param cutoff_val: threshold integer to be passed to cv2.threshold()
        """
        fld_img_value = []
        for idx, i in enumerate(fld_img_sections):
            fld_img1 = i
            fld_img_g = cv2.cvtColor(fld_img1, cv2.COLOR_BGR2GRAY)
            fld_img_gb = cv2.GaussianBlur(fld_img_g, (5, 5), 0)
            # (T, cutoff) = cv2.threshold(fld_img_gb, args['THRESH_BINARY'], 255, cv2.THRESH_BINARY)
            (T, mask) = cv2.threshold(fld_img_gb, cutoff_val, 255, cv2.THRESH_BINARY)
            loc_2 = np.where(255 == mask)
            # for pt in zip(*loc_2[::-1]):
            # fld_img[pt[1], pt[0]] = (0, 0, 255)
            a, b = loc_2
            fld_img_value.append(len(a))

        return fld_img_value

    def generate_binned_map(fld_img_value, fld_img_sect_cords):
        """ Assign bin color to image slices based on the pixel yield values returned from get_pix_count_for_img_sections().
        A legend showing upper limit and color for each bin. This can be more dynamic by creating arguments expressing the
        number of bins to use, the colors to be used, and whether or not to automatically generate a color scale. The logic
        is not automated as evidenced by the redundant elif()

        :param fld_img_value: list of pix yield integers from generate_binned_map()
        :param fld_img_sect_cords: coordinates for generating image slices for binning
        :return field image heat map

        """
        bin_levels = math.ceil(max(fld_img_value) / 8)
        fld_img_value_scale = range(0, max(fld_img_value), bin_levels)
        fld_img_iterate = range(0, len(fld_img_sect_cords) - 1, 1)
        for idx, i in zip(fld_img_iterate, fld_img_sect_cords):
            if fld_img_value[idx] < fld_img_value_scale[1]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (255, 0, 188) #purple
            elif fld_img_value[idx] < fld_img_value_scale[2]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (255, 85, 0) #blue
            elif fld_img_value[idx] < fld_img_value_scale[3]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (145, 255, 0) #light green
            elif fld_img_value[idx] < fld_img_value_scale[4]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (0, 255, 34) #neon green
            elif fld_img_value[idx] < fld_img_value_scale[5]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (0, 239, 255) #yellow
            elif fld_img_value[idx] < fld_img_value_scale[6]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (0, 102, 255) #light orange
            elif fld_img_value[idx] < fld_img_value_scale[7]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (0, 0, 255) #red

    fld_img_sect_cords = get_fld_img_sect_cords(fld_img, bin_size_height, bin_size_width)
    field_img_sections = get_fld_img_sections(fld_img, fld_img_sect_cords, bin_size_height, bin_size_width)
    fld_img_value = get_pix_count_for_img_sections(field_img_sections, cutoff_val)
    generate_binned_map(fld_img_value, fld_img_sect_cords)

    return fld_img
    # bin_legend_colors = ['purple', 'blue', 'light green', 'neon green', 'yellow', 'light orange', 'red']
    # bin_legend = [i for i in zip(fld_img_value_scale, bin_legend_colors)]
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # x, y = 100, 100
    # for i in bin_legend:
    #     cv2.putText(fld_img, str(i), (x, y), font, 5, (0, 0, 0), 5)
    #     x, y = x, y+200

    # cv2.putText(img, str(samp_filename[idx]), (100, 26), font, .5, (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.imwrite(args['outFolder'] + '/WGD_sample_counted_{0}_{1}.jpg'.format(samp_filename[idx], args['THRESH_BINARY']),
    #             img)
    # cv2.imwrite('C:\\Users\\William\\Desktop\\samp_test_out\\output11_'+str(samp_filename[idx]), img)
    # cv2.imwrite('C:\\Users\\William\\Desktop\\output_2017-11-30\\tester_map3.jpg', img)

    def make_binned_heat_map_MODEL(fld_img, bin_size_height, bin_size_width, cutoff_val):
        """ Make heat map by, first: slicing the image into extremely small slices (20 by 20, etc.), second: performing
        yield estimate on each slice and storing that value in a list, third: assigning the section of the original image
        represented by each slice a solid color based on the which bin the pixel yield estimate.

        :param fld_img: composite field image
        :param bin_size_height: height to be used for image slices to be binned
        :param bin_size_width: width to be used for image slices to be banned
        :param cutoff_val: threshold value for pixel yield estimate
        :return: field image heat map with seven bins with distinct colors
        """

    def get_fld_img_sect_cords(fld_img, bin_size_height, bin_size_width):
        """ Take a composite field image and list of coordinates so that the field image can be divided into chunks of
        given height and width. For the purpose of 'binning' each piece and assigning a color.

         :param fld_img: composite field image
         :param bin_size_height: height in pixels of image sections to be 'binned'
         :param bin_size_width: width in pixels of image section to be 'binned'
         :return list of coordinates to slice composite field image for 'binning'
        """

        fld_img_h, fld_img_w, fld_img_c = fld_img.shape
        fld_img_h_s = range(0, fld_img_h, bin_size_height)
        fld_img_w_s = range(0, fld_img_w, bin_size_width)
        fld_img_sect_cords = []
        for y in fld_img_h_s:
            for x in fld_img_w_s:
                fld_img_sect_cords.append((x, y))

        return fld_img_sect_cords

    def get_fld_img_sections(fld_img, fld_img_sect_cords, bin_size_height, bin_size_width):
        """ Use the list of coordinates returned from get_fld_img_sect_cords() to slice image array into individual
        images with proportions equal to bin_size_height and bin_size_width stored in a list

        :param fld_img: composite field image
        :param fld_img_sect_cords: a list of image
        :param bin_size_height:
        :param bin_size_width:
        :return: list of coordinates with which image can be sliced for 'binning'
        """

        fld_img_sections = []
        for i in fld_img_sect_cords:
            fld_img_sections.append(fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width])

        return fld_img_sections

    def get_pix_count_for_img_sections(fld_img_sections, cutoff_val):
        """ Iterate over list of image slices, returned from get_fld_image_sections(), and calculate pixel count for
        that region of the image, then append that value to a list of pixel yield values, then return values.

        :param fld_img_sections: list of sliced field image sections
        :param cutoff_val: threshold integer to be passed to cv2.threshold()
        """
        fld_img_value = []
        for idx, i in enumerate(fld_img_sections):
            fld_img1 = i
            fld_img_g = cv2.cvtColor(fld_img1, cv2.COLOR_BGR2GRAY)
            fld_img_gb = cv2.GaussianBlur(fld_img_g, (5, 5), 0)
            # (T, cutoff) = cv2.threshold(fld_img_gb, args['THRESH_BINARY'], 255, cv2.THRESH_BINARY)
            (T, mask) = cv2.threshold(fld_img_gb, cutoff_val, 255, cv2.THRESH_BINARY)
            loc_2 = np.where(255 == mask)
            # for pt in zip(*loc_2[::-1]):
            # fld_img[pt[1], pt[0]] = (0, 0, 255)
            a, b = loc_2
            x = len(a)
            y = 0.082*x + 48.9
            fld_img_value.append(y)

        return fld_img_value

    def generate_binned_map(fld_img_value, fld_img_sect_cords):
        """ Assign bin color to image slices based on the pixel yield values returned from get_pix_count_for_img_sections().
        A legend showing upper limit and color for each bin. This can be more dynamic by creating arguments expressing the
        number of bins to use, the colors to be used, and whether or not to automatically generate a color scale. The logic
        is not automated as evidenced by the redundant elif()

        :param fld_img_value: list of pix yield integers from generate_binned_map()
        :param fld_img_sect_cords: coordinates for generating image slices for binning
        :return field image heat map

        """
        fld_img_value_scale = range(0, max(fld_img_value), bin_levels)
        fld_img_iterate = range(0, len(fld_img_sect_cords) - 1, 1)
        for idx, i in zip(fld_img_iterate, fld_img_sect_cords):
            if fld_img_value[idx] < fld_img_value_scale[1]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (255, 0, 188) #purple
            elif fld_img_value[idx] < fld_img_value_scale[2]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (255, 85, 0) #blue
            elif fld_img_value[idx] < fld_img_value_scale[3]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (145, 255, 0) #light green
            elif fld_img_value[idx] < fld_img_value_scale[4]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (0, 255, 34) #neon green
            elif fld_img_value[idx] < fld_img_value_scale[5]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (0, 239, 255) #yellow
            elif fld_img_value[idx] < fld_img_value_scale[6]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (0, 102, 255) #light orange
            elif fld_img_value[idx] < fld_img_value_scale[7]:
                fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width] = (0, 0, 255) #red

    fld_img_sect_cords = get_fld_img_sect_cords(fld_img, bin_size_height, bin_size_width)
    field_img_sections = get_fld_img_sections(fld_img, fld_img_sect_cords, bin_size_height, bin_size_width)
    fld_img_value = get_pix_count_for_img_sections(field_img_sections, cutoff_val)
    generate_binned_map(fld_img_value, fld_img_sect_cords)

    return fld_img
    # bin_legend_colors = ['purple', 'blue', 'light green', 'neon green', 'yellow', 'light orange', 'red']
    # bin_legend = [i for i in zip(fld_img_value_scale, bin_legend_colors)]
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # x, y = 100, 100
    # for i in bin_legend:
    #     cv2.putText(fld_img, str(i), (x, y), font, 5, (0, 0, 0), 5)
    #     x, y = x, y+200

    # cv2.putText(img, str(samp_filename[idx]), (100, 26), font, .5, (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.imwrite(args['outFolder'] + '/WGD_sample_counted_{0}_{1}.jpg'.format(samp_filename[idx], args['THRESH_BINARY']),
    #             img)
    # cv2.imwrite('C:\\Users\\William\\Desktop\\samp_test_out\\output11_'+str(samp_filename[idx]), img)
    # cv2.imwrite('C:\\Users\\William\\Desktop\\output_2017-11-30\\tester_map3.jpg', img)


def make_binned_yield_est_map(fld_img, bin_size_height, bin_size_width, cutoff_val):

    """ Make yield heat map by, first: slicing the image into 1 m slices, second: performing
    yield estimate on each slice and storing that value in a list, third: print that estimate on slice and then
    reassemble the image.

    :param fld_img: composite field image
    :param bin_size_height: height to be used for image slices to be binned
    :param bin_size_width: width to be used for image slices to be banned
    :param cutoff_val: threshold value for pixel yield estimate
    :return: field image heat map with seven bins with distinct colors
    """

    def get_fld_img_sect_cords(fld_img, bin_size_height, bin_size_width):
        """ Take a composite field image and list of coordinates so that the field image can be divided into chunks of
        given height and width. For the purpose of 'binning' each piece and assigning a color.

         :param fld_img: composite field image
         :param bin_size_height: height in pixels of image sections to be 'binned'
         :param bin_size_width: width in pixels of image section to be 'binned'
         :return list of coordinates to slice composite field image for 'binning'
        """

        fld_img_h, fld_img_w, fld_img_c = fld_img.shape
        fld_img_h_s = range(0, fld_img_h, bin_size_height)
        fld_img_w_s = range(0, fld_img_w, bin_size_width)
        fld_img_sect_cords = []
        for y in fld_img_h_s:
            for x in fld_img_w_s:
                fld_img_sect_cords.append((x, y))

        return fld_img_sect_cords

    def get_fld_img_sections(fld_img, fld_img_sect_cords, bin_size_height, bin_size_width):
        """ Use the list of coordinates returned from get_fld_img_sect_cords() to slice image array into individual
        images with proportions equal to bin_size_height and bin_size_width stored in a list

        :param fld_img: composite field image
        :param fld_img_sect_cords: a list of image
        :param bin_size_height:
        :param bin_size_width:
        :return: list of coordinates with which image can be sliced for 'binning'
        """

        fld_img_sections = []
        for i in fld_img_sect_cords:
            fld_img_sections.append(fld_img[i[1]:i[1] + bin_size_height, i[0]:i[0] + bin_size_width])

        return fld_img_sections

    def get_pix_count_for_img_sections(fld_img_sections, cutoff_val):
        """ Iterate over list of image slices, returned from get_fld_image_sections(), and calculate pixel count for
        that region of the image, then append that value to a list of pixel yield values, then return values.

        :param fld_img_sections: list of sliced field image sections
        :param cutoff_val: threshold integer to be passed to cv2.threshold()
        """
        fld_img_value = []
        for idx, i in enumerate(fld_img_sections):
            fld_img1 = i
            fld_img_g = cv2.cvtColor(fld_img1, cv2.COLOR_BGR2GRAY)
            fld_img_gb = cv2.GaussianBlur(fld_img_g, (5, 5), 0)
            # (T, cutoff) = cv2.threshold(fld_img_gb, args['THRESH_BINARY'], 255, cv2.THRESH_BINARY)
            (T, mask) = cv2.threshold(fld_img_gb, cutoff_val, 255, cv2.THRESH_BINARY)
            loc_2 = np.where(255 == mask)
            # for pt in zip(*loc_2[::-1]):
            # fld_img[pt[1], pt[0]] = (0, 0, 255)
            a, b = loc_2
            fld_img_value.append(len(a))

        return fld_img_value

    def generate_binned_yield_map(fld_img_value, fld_img_sect_cords):
        """ Assign bin color to image slices based on the pixel yield values returned from get_pix_count_for_img_sections().
        A legend showing upper limit and color for each bin. This can be more dynamic by creating arguments expressing the
        number of bins to use, the colors to be used, and whether or not to automatically generate a color scale.

        :param fld_img_value: list of pix yield integers from generate_binned_map()
        :param fld_img_sect_cords: coordinates for generating image slices for binning
        :return field image heat map

        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for (coords, value) in zip(fld_img_sect_cords, fld_img_value):
            cv2.putText(fld_img, str(value), coords, font, 2, (0, 0, 0), 2)

        return fld_img


    return fld_img

    fld_img_sect_cords = get_fld_img_sect_cords(fld_img, bin_size_height, bin_size_width)
    field_img_sections = get_fld_img_sections(fld_img, fld_img_sect_cords, bin_size_height, bin_size_width)
    fld_img_value = get_pix_count_for_img_sections(field_img_sections, cutoff_val)
    generate_binned_yield_map(fld_img_value, fld_img_sect_cords)

    return fld_img

    # cv2.putText(img, str(samp_filename[idx]), (100, 26), font, .5, (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.imwrite(args['outFolder'] + '/WGD_sample_counted_{0}_{1}.jpg'.format(samp_filename[idx], args['THRESH_BINARY']),
    #             img)
    # cv2.imwrite('C:\\Users\\William\\Desktop\\samp_test_out\\output11_'+str(samp_filename[idx]), img)
    # cv2.imwrite('C:\\Users\\William\\Desktop\\output_2017-11-30\\tester_map3.jpg', img)

def estimate_yield_for_sample_spaces_NPWHERE(in_folder_path, out_folder_path, mrk_tup_L, mrk_tup_H):
    print('Reading sample images...')
    samp_filename = [f for f in listdir(in_folder_path) if isfile(join(in_folder_path, f))]
    sample_image_list = [cv2.imread(in_folder_path + '/' + i) for i in samp_filename]
    print('Estimating yield and writing processed samples......')
    for idx, i in enumerate(sample_image_list):
        img = i
        loc_2 = np.where(np.all(np.logical_and((img > mrk_tup_L), (img < mrk_tup_H)), axis=-1))
        x, y = loc_2
        x = x.tolist()
        y = y.tolist()
        marks = [(x, y) for (x, y) in zip(x, y)]
        for pt in marks:
            img[pt] = (0, 0, 255)
        a, b = loc_2
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(len(a)) + " pixels", (8, 26), font, .75, (0, 242, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(round((-2E-06 * (len(a)) ** 2 + 0.0723 * (len(a)) + 101.67), 3)) + " g/m2", (8, 66), font,
                    .75, (0, 242, 255), 2, cv2.LINE_AA)
        # cv2.putText(img, str(samp_filename[idx]), (100, 26), font, .5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(out_folder_path + '/WGD_sample_counted_{0}_{1}.jpg'.format(samp_filename[idx],
                                                                               'NPWHERE'), img)
    print('File write complete!')


composite_img_IN = cv2.imread("F:\\Drone Images\\2018-05-01_85_85_50_Narrabri\\2018_85_85_50_Narrabri_stitch.jpg")

composite_img_IN = cv2.imread("F:\\Drone Images\\2018-05-01_85_85_50_Narrabri\\2018_85_85_50_Narrabri_stitch_marked.png")

input_folder_path = 'Z:\\Z_drive\\2018-05-01_yield_analysis_Narrabri'
output_folder_path = "Z:\\Z_drive\\2018-05-01_yield_analysis_Narrabri"
composite_id = 'Narrabri_2018-05-01'
raw_marker_color_matches = get_loc(composite_img_IN, (0, 0, 250), (0, 0, 255))
marker_locs = get_one_xy_per_mark(composite_img_IN, raw_marker_color_matches)
#composite_img_IN_marked = draw_samp_circles(composite_img_IN, marker_locs, 155)
composite_img_IN_marked = draw_samp_rectangle(composite_img_IN, marker_locs, 36, 36)
#final_samples = get_circle_samps(composite_img_IN, marker_locs, 155)
final_samples = get_rectangle_samps(composite_img_IN, marker_locs, 36, 36)
write_samps(final_samples, composite_img_IN_marked, composite_id, output_folder_path)
#estimate_yield_for_sample_spaces(input_folder_path, 160, output_folder_path)
estimate_yield_for_sample_spaces_NPWHERE(input_folder_path, output_folder_path, (200, 100, 100), (256, 256, 256))

yield_map = make_binned_yield_est_map(composite_img_IN, 192, 144, 200)

cv2.imwrite('Z:\\Z_drive\\2018-04-30_Yield_analysis_Narrabri\\test.jpg', composite_img_IN_marked)

img_T = make_binned_heat_map(img_T, 150, 150, 190)


#estimate_yield_for_sample_spaces_NPWHERE(input_folder_path, output_folder_path, (190,190,190),(256,256,256))
# ########################################################################################################################
# ########################################################################################################################
#
# heat_map1 = make_binned_heat_map(img_T, 20, 20, 200)
# cv2.imwrite('C:\\Users\\dodge\\Desktop\\testerOUT8.JPG', fld_img)
#
#
# def get_mrk_tuple(marker, mask = None):
#     """ This will return a tuple of the value between 0 and 256 that appears the most for each channel. The tuple
#     is returned in B, G, R order. The tuple represents the value for each color chanel that appears in the marker
#     sample image most. When these values are taken together as a single tuple they may actually form a color that is
#     not most common but the frequency of individual values for each color channel will be returned.
#
#     :param marker: cropped marker image containing color to be extracted
#     :param mask: mask to be applied if desired
#     :return tuple containing most frequent value for each color chanel in image
#
#     """
#     marker = cv2.GaussianBlur(marker, (15, 15), 0)
#     chans_m = cv2.split(marker)
#     max_val = []
#
#     for chan in chans_m:
#         hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
#         hist = hist.argmax(axis=0)
#         max_val.append(hist[0])
#
#     return tuple(max_val)
# def write_samp_files(fld_img, smp_bounds, out_path):
#     fin_samp = [fld_img[i[0][0]:i[0][1], i[1][0]:i[1][1]] for i in smp_bounds]
#     os.chdir(out_path)
#     for idx, sample in enumerate(fin_samp):
#         # cv2.imwrite('WGDpixHrvst_0.01_samples_' + str(idx) + '_' + str(args['field']) + '_thrsh_' +
#                     #str(args['thresh']) + '_' + str(args['thresh_2']) + '.jpg', sample)
#         cv2.imwrite('samp2017-12-10_'+str(idx)+'.jpg', fin_samp[idx])
#
#     print('sample image file write complete')
# def get_filepaths(dir_path):
#     file_path_list = os.listdir(dir_path)
#
#     return file_path_list
# def draw_samps_trapezoid(fld_img, marks_1):
#     h, w = (40, 40)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     for idx, i in enumerate(marks_1):
#         points = np.array([[i[1]-40, i[0]+20], [i[1]+40, i[0]+20], [i[1]+100, i[0]-300], [i[1]-100, i[0]-300]])
#         cv2.polylines(fld_img, [points], 1, (0, 255, 0), 8)
#         cv2.putText(fld_img, str(idx), (i[1] + w, i[0] - 10), font, 1.5, (255, 51, 255), 4)
#
#     return fld_img


# ap = argparse.ArgumentParser()
#
# ap.add_argument('-f', '--field', required = True,
#                             help = 'path to field image')
# ap.add_argument('-mL', '--marker_tuple_low', required = True,
#                             help = 'low end of marker tuple range', type=int, nargs='+')
# ap.add_argument('-mH', '--marker_tuple_high', required = True,
#                             help = 'high end of marker tuple range', type=int, nargs='+')
# ap.add_argument('-o', '--outFolder', required = True,
#                             help = 'output folder path')
#
# img_1 = cv2.imread('C:\\Users\\dodge\\Desktop\\2017_cropped_composites_beltwide_experiment\\75_75_30_2017-11-16_cropped.jpg')
# img_2 = cv2.imread('C:\\Users\\dodge\\Desktop\\2017_cropped_composites_beltwide_experiment\\85_85_30_2017-11-14_cropped.jpg')
#img_T = cv2.imread('C:\\Users\\William\\Desktop\\65_65_30_2017-11-17_cropped.jpg')
#field_image = img_T
#img_T_comp = cv2.imread('C:\\Users\\dodge\\Desktop\\75_75_20_2017-11-17_croppedTESTER.jpg')
#field_image = img_T
#img_T = cv2.imread('C:\\Users\\dodge\\Desktop\\tester1.JPG')
#field_image = img_T
#img = cv2.imread('C:\\Users\\William\\Desktop\\test_2.jpg')
#marker = cv2.imread('C:\\Users\\William\\Desktop\\marker_4.jpg')
# args = vars(ap.parse_args())
#
# img = cv2.imread(args['field'])
#######################################################
#######################################################
# marks_loc_1 = get_loc_2(img_T, marker_tuple_low, marker_tuple_high)
# marks_loc_1 = get_one_xy_per_mark(img_T, marks_loc_1)
# draw_samps_circle(img_T, marks_loc_1)
#
# samples_out = get_circle_samps(img_T, marks_loc_1)
#
#
# directory_path = 'C:\\Users\\William\\Desktop\\2017_croped_TESTOUT4\\'
#
# file_path_list = get_filepaths(directory_path)
#
# marker_tuple_low = (10, 100, 225)
# marker_tuple_high = (90, 180, 256)
#
# for idx, img in enumerate(file_path_list):
#     field_image = cv2.imread(directory_path + str(img))
#     marks_all_pixels = get_loc_2(field_image, marker_tuple_low, marker_tuple_high)
#     marks_one_xy_per_mark = get_one_xy_per_mark(field_image, marks_all_pixels)
#     draw_samps_circle(field_image, marks_one_xy_per_mark)
#     cv2.imwrite(directory_path + str(img) + '_marked_output.jpg', field_image)
#
# cv2.imwrite('C:\\Users\\William\\Desktop\\circle_out_8.jpg', field_image)
#
#
#
#
#
