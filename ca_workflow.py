from aean import CropAnalyzer
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

# current_set_path = '/home/will/drone-code/2018-05-30_75_75_20_mauricio_odm_aoms_extracted.tif'
current_set_path = '/home/will/drone-code/2018-06-03_75_75_20_mauricio_odm_aoms_extracted.tif'

out_put_dir = 'out_put'


analyzer = CropAnalyzer(aom_set_path=current_set_path,
                        num_aom_rows=56,
                        num_aom_tiers=25,
                        germ_id_set_path="/home/will/drone-code/mauricio_field_map_ID_only_UPDATED.csv",
                        output_folder_path="/home/will/drone-code/{0}".format(out_put_dir),
                        analysis_id="2018-06-03_75_75_20_mauricio_odm_aoms_extracted.tif",
                        )

analyzer.germ_id_modify()
analyzer.read_aom_set()
analyzer.get_set_contours_centers(thresh_val=255, area_filt=5000)
analyzer.group_contours()
analyzer.organize_centers()
analyzer.make_row_tier_tags()
analyzer.make_sample_images()

del analyzer.aom_row_tier_tags[1344:1346]

analyzer.put_id_on_sample_map()
analyzer.make_green_diff_set()

analyzer.count_plants(thresh_val=100, area_filt=12)
analyzer.make_plant_count_df()

analyzer.plant_count_data[0].to_csv('/home/will/drone-code/{0}/plant_count_data_countour.csv'.format(out_put_dir), index=False)
analyzer.plant_count_data[1].to_csv('/home/will/drone-code/{0}/plant_count_data_green_pix.csv'.format(out_put_dir), index=False)


analyzer.write_sample_map()

analyzer.samples_dir = 'unmarked_samples'
analyzer.write_samples()

analyzer.green_diff_dir = 'green_diff'
analyzer.write_green_diff_samples()

analyzer.plants_counted_dir = 'plants_counted'
analyzer.write_counted_plant_samples()

