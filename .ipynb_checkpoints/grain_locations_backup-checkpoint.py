import pandas as pd
import numpy as np
import pims
import h5py
import matplotlib.pyplot as plt
import xarray as xr
import pathlib
import cv2 as cv
import tqdm
tqdm.tqdm.pandas()
import os


class grain_locations(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.pims_path = file_path
        self.path = file_path.parent

        if self.path.stem == 'manta':
            self.name = str(file_path.parent.parent.stem) + '_locs.h5'
        elif self.path.stem == 'edgertronic':
            self.name = str(self.pims_path.stem) + '_locs.h5'

        self.file_name = self.path / self.name

        self.info = vid_info
        if os.path.isfile(str(self.file_name)) is True:
            self.dt = 1 / self.info['frame_rate'] # time between frames
            self.pix_to_mm = 4.95 / self.get_attr('mean_radius')
        else:
            self.pix_to_mm = np.nan


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
                dset = dset[np.where((dset[:,3] >= frange[0]) & (dset[:,3] < frange[1]))[0],:]
                return xr.Dataset({
                        'time': ('frame', dset[:,3]*self.dt),
                        'radius': ('frame', dset[:,2]),
                        'x_pix': ('frame', dset[:,1]),
                        'x_mm': ('frame', dset[:,1]*self.pix_to_mm),
                        'y_pix': ('frame', dset[:,0]),
                        'y_mm': ('frame', dset[:,0]*self.pix_to_mm)
                        },
                    coords={
                        'frame': ('frame', dset[:,3].astype(int)),
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
