import numpy as np
import pims
import os
import pandas as pd
import trackpy
from scipy import ndimage
import trackpy.predict as tp_predict
import grain_locations
import bed_surfaces
import pathlib
import xarray as xr
import tqdm
tqdm.tqdm.pandas()


class grain_tracks(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.info = vid_info
        self.pims_path = file_path
        self.path = file_path.parent

        if self.path.stem == 'manta':
            self.name = str(file_path.parent.parent.stem) + '_tracks.h5'
        elif self.path.stem == 'edgertronic':
            self.name = str(self.pims_path.stem) + '_tracks.h5'

        self.file_name = self.path / self.name
        self.locations = grain_locations.grain_locations(self.pims_path, vid_info)
        self.bed = bed_surfaces.bed_surfaces(self.pims_path, vid_info)

    # method to output dataset
    def get(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        xf = xr.open_dataset(self.file_name).set_index(ind=['frame', 'particle'])
        return xf.where((xf.frame>=frange[0]) & (xf.frame<frange[1]), drop=True)

    def calculate(self, frange=None, batch_size=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        # open grain locations file
        rings = self.locations.get(frange)

        if rings is None:
            print('Need to calculate grain locations...')
            return

        particles = rings.to_dataframe().reset_index(level=['frame']).rename(index=str, columns={'radius': 'mass', 'x_pix': 'x', 'y_pix': 'y'}).drop(columns=['x_mm', 'y_mm', 'time'])
        d = particles.mass.mean()*2
        xf_bed = self.bed.get(frange)
        pix2mm = self.locations.pix_to_mm

        print('Imprinting bed location...')
        particles = particles.groupby('frame').progress_apply(lambda x: self.particle_activity(x, d/2, pix2mm, xf_bed))

        print('Tracking particles...')
        if self.path.stem == 'edgertronic':
            # pred = trackpy.predict.NearestVelocityPredict()
            # p_tracks = pred.link_df(particles, search_range=1.5*d, adaptive_stop=0.5, adaptive_step=0.9, memory=5)
            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks_load = pred.link_df(particles[particles.activity == 1], search_range=0.9*d, adaptive_stop=0.5, adaptive_step=0.98, memory=10)

            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks_bed = pred.link_df(particles[particles.activity == 0], search_range=d/5, adaptive_stop=0.5, adaptive_step=0.98, memory=1)

            p_tracks_bed.particle += p_tracks_load.particle.max() + 1
            p_tracks = pd.concat([p_tracks_load, p_tracks_bed])

        elif self.path.stem == 'manta':
            # pred = trackpy.predict.NearestVelocityPredict()
            # p_tracks = pred.link_df(particles, search_range=d/2, adaptive_stop=0.5, adaptive_step=0.9, memory=5)
            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks_load = pred.link_df(particles[particles.activity == 1], search_range=2*d, adaptive_stop=0.5, adaptive_step=0.98, memory=10)

            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks_bed = pred.link_df(particles[particles.activity == 0], search_range=d/10, adaptive_stop=0.5, adaptive_step=0.98, memory=10)

            p_tracks_bed.particle += p_tracks_load.particle.max() + 1
            p_tracks = pd.concat([p_tracks_load, p_tracks_bed])

        print('Smoothing tracks...')
        p_tracks = p_tracks.rename(index=str, columns={'x': 'x_pix', 'y': 'y_pix'})
        results = p_tracks.groupby('particle').progress_apply(self.sub_pxl_res)

        # save particle velocities
        if os.path.isfile(str(self.file_name)) is True:
            os.remove(str(self.file_name))

        xr.Dataset({
            'radius': ('ind', results.mass.values),
            'x_pix': ('ind', results.x_pix.values),
            'y_pix': ('ind', results.y_pix.values),
            'dy_pix': ('ind', results.dy_pix.values),
            'x_sub_pix': ('ind', results.x_sub_pix.values),
            'y_sub_pix': ('ind', results.y_sub_pix.values),
            'dy_sub_pix': ('ind', results.dy_sub_pix.values),
            'activity': ('ind', results.activity.values),
            'fractional_activity': ('ind', results.fractional_activity.values)
            },
            coords={
                'frame': ('ind', results.frame.values),
                'particle': ('ind', results.particle.values)
            }).to_netcdf(self.file_name)

    def sub_pxl_res(self, group):
        group.sort_values(by='frame', inplace=True)

        activity = group.activity.mean()
        group['fractional_activity'] = activity

        N = group.frame.size
        if activity > 0.8:
            n = 3
        else:
            n = 51

        if N < n:
            group['x_sub_pix'] = np.nan
            group['y_sub_pix'] = np.nan
            group['dy_sub_pix'] = np.nan
        else:
            group['x_sub_pix'] = group.x_pix.interpolate(limit=11).rolling(window=n, center=True, win_type='nuttall', min_periods=1).mean()
            group['y_sub_pix'] = group.y_pix.interpolate(limit=11).rolling(window=n, center=True, win_type='nuttall', min_periods=1).mean()
            group['dy_sub_pix'] = group.dy_pix.interpolate(limit=11).rolling(window=n, center=True, win_type='nuttall', min_periods=1).mean()

        # dx = .5
        # x_T = group.x.values
        # y_T = group.y.values
        # x_m = np.zeros(N)
        # y_m = np.zeros(N)
        #
        # # handle end cases (make this fancier)
        # x_m[:n] = x_T[:n]
        # x_m[-n:] = x_T[-n:]
        # y_m[:n] = y_T[:n]
        # y_m[-n:] = y_T[-n:]
        #
        # # do middle
        # for k in range(n,N-n+1):
        #     x = x_T[k-n:k+n]
        #     y = y_T[k-n:k+n]
        #     X, Y = np.meshgrid(np.arange(x.min(),x.max()+dx, dx), np.arange(y.min(), y.max()+dx, dx))
        #     loc = np.zeros_like(X)
        #     for i in range(2*n):
        #         loc += np.where(X == x[i], 1, 0) * np.where(Y == y[i], 1, 0)
        #
        #     y_m_t, x_m_t = ndimage.measurements.center_of_mass(loc)
        #     x_m[k] = x.min()+x_m_t*dx
        #     y_m[k] = y.min()+y_m_t*dx
        #
        # group['x_sub_pix'] = self.rolling_mean(x_m, window_len=n)
        # # group['x_sub_pix'] = self.rolling_mean(group.x, window_len=n)
        # group['y_sub_pix'] = self.rolling_mean(y_m, window_len=n)
        # # group['y_sub_pix'] = self.rolling_mean(group.y, window_len=n)
        # group['dy_sub_pix'] = np.nan#self.rolling_mean(group.y, window_len=n)

        return group

    def particle_activity(self, group, d, sf, xf_bed):
        bb = xf_bed.sel(frame=group.frame.values[0]).y_pix_bed.values + d
        y_bed = bb[group.x.values.astype(int)]
        y_particle = group.y.values
        dy = y_bed - y_particle
        group['dy_pix'] = dy
        group['activity'] = np.where(dy >= 0, 1, 0)
        return group
