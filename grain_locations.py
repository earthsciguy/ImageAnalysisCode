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
import skimage as sk
import os
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
import ridge_directed_ring_detector as ring_detector
import scipy.spatial.distance as dist


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
        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        self.dt = 1 / self.info['frame_rate'] # time between frames
        if os.path.isfile(str(self.file_name)) is True:
            self.pix_to_mm = self.get_pix_to_mm()
        else:
            self.pix_to_mm = np.nan
            self.calculate([0,1000])
            self.pix_to_mm = self.get_pix_to_mm()


    # method to output dataset
    def get(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        xf = xr.open_dataset(self.file_name)
        xf['x_mm'] = xf.x_pix * self.pix_to_mm
        xf['y_mm'] = xf.y_pix * self.pix_to_mm

        return xf.where((xf.frame>=frange[0]) & (xf.frame<frange[1]), drop=True)


    def get_pix_to_mm(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        xf = xr.open_dataset(self.file_name)

        try:
            self.radius = xf.attrs['radius']

        except:
            self.radius = xf.radius.mean().values
            xf = xf.assign_attrs({'radius': self.radius})
            os.remove(str(self.file_name))
            xf.to_netcdf(self.file_name)

        return 4.95 / (2*self.radius)


    def ridge_hough_rings(self, img, params={'rc': 20, 'circle_thresh': 2 * np.pi * 0.1, 'sigma': 1, 'curv_thresh': -50, 'vote_thresh': 5, 'dr': 1, 'eccentricity': 0}, dr_filter=3):

        ridge_hough = ring_detector.RidgeHoughTransform(cv.bitwise_not(cv.cvtColor(img, cv.COLOR_BGR2GRAY)).astype(np.float32))
        default_params = ridge_hough.params.copy()

        r = (params['rc']-params['dr'], params['rc']+params['dr'])
        ridge_hough.params['circle_thresh'] = params['circle_thresh']
        ridge_hough.params['sigma'] = params['sigma']
        ridge_hough.params['curv_thresh'] = params['curv_thresh']
        ridge_hough.params['Rmin'] = r[0]
        ridge_hough.params['Rmax'] = r[1]
        ridge_hough.params['vote_thresh'] = params['vote_thresh']
        ridge_hough.params['dr'] = params['dr']
        ridge_hough.params['eccentricity'] = params['eccentricity']

        ridge_hough.img_preprocess()
        ridge_hough.rings_detection()
        ring = ridge_hough.output['rings_subpxl']


        # filter overlapping particles
        dr = ring[:,2].mean()/dr_filter
        # find distances between all particles
        distances = dist.cdist(ring[:,:2], ring[:,:2])
        # find particles that are closer than dr and are not themselves
        x, y = np.where(distances<dr)
        d_ring = x[x!=y]
        # filter out those close particles and take average of their location
        close_rings = ring[d_ring, :]
        new_rings = []
        while close_rings.size > 0:
            close_ps = dist.cdist(close_rings[0,:2].reshape((1,2)), close_rings[:,:2])
            xp, yp = np.where(close_ps<dr)
            new_rings.append(np.mean(close_rings[yp,:], axis=0))
            close_rings = close_rings[np.delete(np.arange(close_rings.shape[0]), yp)]
        new_rings = np.array(new_rings)
        # delete double particles from ring array, and replace them with averaged locations
        ind = np.arange(ring.shape[0])
        ind = np.delete(ind, d_ring)
        ring = ring[ind,:]
        if new_rings.size > 0:
            ring = np.vstack([ring, new_rings])

        return ring


    def calculate(self, frange=None, batch_size=None):
        params = {
            'gb_s2_e1':  {'vid0': {'rc': 21, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 5, 'dr': 1, 'eccentricity': 0}},
            'gb_s2_e2':  {'vid0': {'rc': 25, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s2_e3':  {'vid0': {'rc': 25, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s2_e4':  {'vid0': {'rc': 21, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 2, 'eccentricity': 0}},
            'gb_s2_e5':  {'vid0': {'rc': 21, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s2_e6':  {'vid0': {'rc': 20, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e1':  {'vid0': {'rc': 17, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e2':  {'vid0': {'rc': 18, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e3':  {'vid0': {'rc': 17, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e4':  {'vid0': {'rc': 17, 'circle_thresh': 2*np.pi*0.1,  'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e5':  {'vid0': {'rc': 12, 'circle_thresh': 2*np.pi*0.18, 'sigma': 1,  'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e6':  {'vid0': {'rc': 11, 'circle_thresh': 2*np.pi*0.14, 'sigma': .1, 'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e7':  {'vid0': {'rc': 11, 'circle_thresh': 2*np.pi*0.15, 'sigma': .1, 'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e8':  {'vid0': {'rc': 11, 'circle_thresh': 2*np.pi*0.15, 'sigma': .1, 'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e9':  {'vid0': {'rc': 11, 'circle_thresh': 2*np.pi*0.15, 'sigma': .1, 'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e10': {'vid0': {'rc': 11, 'circle_thresh': 2*np.pi*0.15, 'sigma': .1, 'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e11': {'vid0': {'rc': 11, 'circle_thresh': 2*np.pi*0.15, 'sigma': .1, 'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e12': {'vid0': {'rc': 11, 'circle_thresh': 2*np.pi*0.15, 'sigma': .1, 'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
            'gb_s3_e13': {'vid0': {'rc': 11, 'circle_thresh': 2*np.pi*0.15, 'sigma': .1, 'curv_thresh': -100, 'vote_thresh': 6, 'dr': 1, 'eccentricity': 0}},
                }
        if frange == None:
            frange = (0, np.int(self.info['frame_count']))

        x_cent = []
        y_cent = []
        radius = []
        frame = []
        for frame_num in tqdm.tqdm(range(frange[0], frange[1])):
            rings = self.ridge_hough_rings(self.frames[frame_num], params[self.path.parent.stem]['vid0'])
            x_cent.append(rings[:, 1])
            y_cent.append(rings[:, 0])
            radius.append(rings[:, 2])
            frame.append(frame_num * np.ones(rings.shape[0]))

        xr.Dataset({
            'time': ('frame', np.hstack(frame)*self.dt),
            'x_pix': ('frame', np.hstack(x_cent)),
            'y_pix': ('frame', np.hstack(y_cent)),
            'radius': ('frame', np.hstack(radius)),
            },
            coords={
                'frame': ('frame', np.hstack(frame)),
            }).to_netcdf(self.file_name)


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
