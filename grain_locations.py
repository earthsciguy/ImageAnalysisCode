import pandas as pd
import numpy as np
import os
import pims
import h5py
import matplotlib.pyplot as plt
import xarray as xr
import pathlib
import cv2 as cv

class grain_locations(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.pims_path = file_path
        self.path = file_path.parent

        self.name = str(file_path.parent.parent.stem) + '_locs.h5'
        self.file_name = self.path / self.name
        # self.make_group()

        self.info = vid_info
        self.dt = 1./self.info['frame_rate'] # time between frames
        self.pix_to_mm = 4.95 / self.get_attr('mean_radius')


    def make_group(self):
        # open given hdf5 file, file is safely closed when with statement ends
        with h5py.File(self.file_name, 'a') as f:
            print(list(f.keys()))
            dset = f['/grain_locs'][:,0]
            self.make_dataset_attr('start_frame', int(dset.min()))
            self.make_dataset_attr('end_frame', int(dset.max()))
            self.make_dataset_attr('mean_radius', np.nanmean(f['/grain_locs'][:,1]))

    # method to add attributes to given dataset
    def make_dataset_attr(self, attribute_title, attribute_value):
        # open file
        with h5py.File(self.file_name, 'r+') as f:
            dset = f['/grain_locs']
            dset.attrs[attribute_title] = attribute_value

    # method to get all attributes for a dataset
    def get_attrlist(self):
        # open file
        with h5py.File(self.file_name, 'r+') as f:
            dset = f['/grain_locs']
            return [(name, val) for name, val in dset.attrs.items()]

    # method to get a specific attribute value for dataset
    def get_attr(self, attribute_name):
        # open file
        with h5py.File(self.file_name, 'r+') as f:
            dset = f['/grain_locs']
            # ask for attribute, will return 'None' if the attribute does not exist
            return dset.attrs.get(attribute_name)

    # method to output dataset
    def get(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        # open given hdf5 file, file is safely closed when with statement ends
        with h5py.File(self.file_name, 'r+') as f:
                dset = f['/grain_locs'][...]  # read dataset from hdf5 file
                dset = dset[np.where((dset[:,0] >= frange[0]) & (dset[:,0] < frange[1]))[0],:]
                self.dt = 1/130
                self.pix_to_mm = 2
                return xr.Dataset({
                        'time': ('frame', dset[:,0]*self.dt),
                        'radius': ('frame', dset[:,1]),
                        'x_pix': ('frame', dset[:,2]),
                        'x_mm': ('frame', dset[:,2]*self.pix_to_mm),
                        'y_pix': ('frame', dset[:,3]),
                        'y_mm': ('frame', dset[:,3]*self.pix_to_mm)
                        },
                    coords={
                        'frame': ('frame', dset[:,0].astype(int)),
                        })

    def see_frame(self, frame_num):
        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        img = self.frames[frame_num]

        rings = self.get(frange=(frame_num, frame_num+1))

        if rings is not None:
            # r = int(rings.radius.mean())
            for ii in range(rings.radius.values.size):
                # draw the center of the circle
                cv.circle(img, (int(rings.x_pix[ii].values),int(rings.y_pix[ii].values)), rings.radius[ii].values, (0,0,0), 1, cv.LINE_AA)

        fig = plt.figure(figsize=(15,15))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

# i = 1
# file_name = pathlib.Path('/Users/ericdeal/Dropbox (MIT)/3_postdoc/projects/sed_transport/1_data/0_main_feed_exp_data/_2017_exps/_data/glass_beads/exp_transport_stage_%i/manta/glass_beads_feed_n3p3_manta_record_130_playback_32.5.mp4'%i)
# obj = grain_locations(file_name)
