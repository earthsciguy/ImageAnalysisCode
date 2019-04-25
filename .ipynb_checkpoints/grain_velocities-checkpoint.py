import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import pims
import pathlib
import h5py
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tqdm
tqdm.tqdm.pandas()
import grain_locations
import grain_tracks
import bed_surfaces


class grain_velocities(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None, rotation=0):
        self.info = vid_info
        self.pims_path = file_path
        self.path = file_path.parent

        if self.path.stem == 'manta':
            self.name = str(file_path.parent.parent.stem) + '_velocities.h5'
        elif self.path.stem == 'edgertronic':
            self.name = str(self.pims_path.stem) + '_velocities.h5'

        self.file_name = self.path / self.name
        self.locs = grain_locations.grain_locations(self.pims_path, vid_info)
        self.tracks = grain_tracks.grain_tracks(self.pims_path, vid_info)
        self.bed = bed_surfaces.bed_surfaces(self.pims_path, vid_info)
        self.theta = rotation # degrees to rotate bed
        self.min_track_life = 5


    # method to output dataset
    def get(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        xf = xr.open_dataset(self.file_name).set_index(ind=['frame', 'particle'])
        return xf.where((xf.frame>=frange[0]) & (xf.frame<frange[1]), drop=True)


    def velocity_calc(self, df):
        df.sort_values(by='frame', inplace=True)

        if df.index.size <= self.min_track_life:
            for key in ['vx_pix', 'vy_pix', 'vx_sub_pix', 'vy_sub_pix']:
                df[key] = np.nan
            return df

        else:
            # position of particle along track
            pos = {'x_pix': df.x_pix.values, 'y_pix': df.y_pix.values, 'x_sub_pix': df.x_sub_pix.values, 'y_sub_pix': df.y_sub_pix.values}
            # Calculate time difference between track locations
            dt = np.diff(df.time.values)

            v = {}
            for key in pos:
                # print(key)
                # create empty vectors to be filled
                v[key] = np.zeros_like(pos[key])
                ############### Centered difference order 4
                 # fill first and last elements
                v[key][0] = (pos[key][1] - pos[key][0]) / dt[0]
                v[key][1] = (pos[key][2] - pos[key][1]) / dt[1]
                v[key][-2] = (pos[key][-2] - pos[key][-3]) / dt[-2]
                v[key][-1] = (pos[key][-1] - pos[key][-2]) / dt[-1]
                # find velocity for middle elements
                for ii in range(2,dt.size-1):
                    v[key][ii] = (-pos[key][ii+2] + 8*pos[key][ii+1] - 8*pos[key][ii-1] + pos[key][ii-2]) / (12*dt[ii-2:ii+2].mean())

                # for ii in range(2,dt.size-1):
                #     v[key][ii] = (pos[key][ii+1] - pos[key][ii-1]) / (2*dt[ii-2:ii+2].mean())

                # save to original dataframe
                df['v'+key] = v[key]
            return df


    def calculate(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        print('Calculating grain velocities...')

        # open grain locations file
        df = self.tracks.get(frange).reset_index('ind').to_dataframe().reset_index()
        df['time'] = df.frame.values * self.locs.dt
        results = df.groupby(['particle']).progress_apply(self.velocity_calc)

        # save particle velocities
        if os.path.isfile(str(self.file_name)) is True:
            os.remove(str(self.file_name))

        n = 11
        xr.Dataset({
            'time': ('ind', results.time.values),
            'radius': ('ind', results.radius.values),
            'radius_mm': ('ind', results.radius.values * self.locs.pix_to_mm),
            'radius_sub': ('ind', results.radius_sub.values),
            'radius_sub_mm': ('ind', results.radius_sub.values * self.locs.pix_to_mm),
            'x_pix': ('ind', results.x_pix.values),
            'x_mm': ('ind', results.x_pix.values * self.locs.pix_to_mm),
            'y_pix': ('ind', results.y_pix.values),
            'y_mm': ('ind', results.y_pix.values * self.locs.pix_to_mm),
            'dy_pix': ('ind', results.dy_pix.values),
            'dy_mm': ('ind', results.dy_pix.values * self.locs.pix_to_mm),
            'x_sub_pix': ('ind', results.x_sub_pix.values),
            'x_sub_mm': ('ind', results.x_sub_pix.values * self.locs.pix_to_mm),
            'y_sub_pix': ('ind', results.y_sub_pix.values),
            'y_sub_mm': ('ind', results.y_sub_pix.values * self.locs.pix_to_mm),
            'dy_sub_pix': ('ind', results.dy_pix.values),
            'dy_sub_mm': ('ind', results.dy_pix.values * self.locs.pix_to_mm),
            'vx_pix': ('ind', results.vx_pix.values),
            'vx_mm': ('ind', results.vx_pix.values * self.locs.pix_to_mm),
            'vy_pix': ('ind', results.vy_pix.values),
            'vy_mm': ('ind', results.vy_pix.values * self.locs.pix_to_mm),
            'vx_sub_pix': ('ind', results.vx_sub_pix.rolling(window=n, win_type='hamming', center=True, min_periods=n).mean().values),
            'vx_sub_mm': ('ind', results.vx_sub_pix.rolling(window=n, win_type='hamming', center=True, min_periods=n).mean().values * self.locs.pix_to_mm),
            'vy_sub_pix': ('ind', results.vy_sub_pix.rolling(window=n, win_type='hamming', center=True, min_periods=n).mean().values),
            'vy_sub_mm': ('ind', results.vy_sub_pix.rolling(window=n, win_type='hamming', center=True, min_periods=n).mean().values * self.locs.pix_to_mm),
            'activity': ('ind', results.activity.values),
            'fractional_activity': ('ind', results.fractional_activity.values)
            },
            coords={
                'frame': ('ind', results.frame.values),
                'particle': ('ind', results.particle.values)
            }).to_netcdf(self.file_name)


    def see_frame(self, frange, ret=None, smoothed=True):
        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        bed = self.bed.get(frange=frange)
        rings = self.get(frange=frange)
        s1 = 256
        s2 = 100

        imgs = np.zeros((int(frange[1]-frange[0]), int(self.info['vertical_dim']), int(self.info['horizontal_dim']), 3))
        for ii, frame_num in enumerate(tqdm.tqdm(range(frange[0], frange[1]))):
            img = self.frames[frame_num]

            y = bed.sel(frame=frame_num).y_pix_bed_lin.values
            # draw bed surface
            x = np.arange(y.size)
            pts_to_draw = np.array([[np.round(x[ii]*s1).astype(np.int), np.round(y[ii]*s1).astype(np.int)] for ii in range(x.size)])
            # draw bed line on image
            cv.polylines(img, np.int32([pts_to_draw]), False, (0, 0, 0), 2, cv.CV_AA, shift=8)
            for row, ring in rings.sel(frame=frame_num).groupby('particle'):
                # draw the center of the circle
                if smoothed == False:
                    try:
                        x = int(ring.x_pix.values*s1)
                        y = int(ring.y_pix.values*s1)
                        vx = int(ring.x_pix.values*s1 + ring.vx_pix.values*s1/s2)
                        vy = int(ring.y_pix.values*s1 + ring.vy_pix.values*s1/s2)
                        rad = int(ring.radius.values*s1)
                    except:
                        continue
                elif smoothed == True:
                    try:
                        x = int(ring.x_sub_pix.values*s1)
                        y = int(ring.y_sub_pix.values*s1)
                        # vx_mag = np.log10(np.abs(ring.vx_sub_pix.values))
                        # vy_mag = np.log10(np.abs(ring.vy_sub_pix.values))
                        # vx_dir = ring.vx_sub_pix.values / np.abs(ring.vx_sub_pix.values)
                        # vy_dir = ring.vy_sub_pix.values / np.abs(ring.vy_sub_pix.values)
                        vx = int(ring.x_sub_pix.values*s1 + ring.vx_sub_pix.values*s1/s2)
                        vy = int(ring.y_sub_pix.values*s1 + ring.vy_sub_pix.values*s1/s2)
                        rad = int(ring.radius_sub.values*s1/3)
                    except:
                        continue

                color = (0,0,0)
                cv.circle(img, (x, y), 3*s1, color, -1, cv.LINE_AA, shift=8)
                cv.arrowedLine(img, (x, y), (vx, vy), color, 2, cv.LINE_AA, shift=8)
                # cv.circle(img, (x, y), rad, color, 1, cv.LINE_AA, shift=8)

            imgs[ii,:,:,:] = img

        if ret == 'image':
            return imgs
        else:
            plt.imshow(imgs[0].astype(np.uint8))
            plt.axis('off')
            plt.tight_layout()
            plt.show()


    def make_movie(self, frange=None, smoothed=True):
        if frange == None:
            frange = (0, self.info['frame_count'])

        fps = 30
        num_images = frange[1] - frange[0] - 1
        duration = num_images/fps
        imgs = self.see_frame(frange, ret='image', smoothed=smoothed)
        def make_frame(t):
            return imgs[int(t*fps)]

        animation = mpy.VideoClip(make_frame, duration=duration)
        if smoothed == False:
            animation.write_videofile(str(self.file_name.parent / self.file_name.stem) + '_output_.mp4', fps=fps) # export as video
        elif smoothed == True:
            animation.write_videofile(str(self.file_name.parent / self.file_name.stem) + '_output_smoothed.mp4', fps=fps) # export as video
