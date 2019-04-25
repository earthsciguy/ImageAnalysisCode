import numpy as np
import pims
import pandas as pd
import trackpy
from scipy import ndimage
import trackpy.predict as tp_predict
import pathlib
import xarray as xr
import tqdm
tqdm.tqdm.pandas()
import os
import grain_locations
import bed_surfaces


class grain_tracks(object):

    # init method run when instance is created
    def __init__(self, locs, bed, file_path=None, vid_info=None):
        self.info = vid_info
        self.pims_path = file_path
        self.path = file_path.parent

        if self.path.stem == 'manta':
            self.name = str(file_path.parent.parent.stem) + '_tracks.h5'
        elif self.path.stem == 'edgertronic':
            self.name = str(self.pims_path.stem) + '_tracks.h5'

        self.file_name = self.path / self.name
        # self.locations = grain_locations.grain_locations(self.pims_path, vid_info)
        self.locations = locs
        # self.bed = bed_surfaces.bed_surfaces(self.pims_path, vid_info)
        self.bed = bed
        self.fps = vid_info['frame_rate']


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

        particles = rings.to_dataframe().reset_index(level=['frame']).rename(index=str, columns={'radius': 'mass', 'x_pix': 'x', 'y_pix': 'y'}).drop(columns=['x_mm', 'y_mm', 'time']).dropna()

        self.d = particles.mass.mean()

        xf_bed = self.bed.get(frange)
        pix2mm = self.locations.pix_to_mm

        print('Imprinting bed location...')
        particles = particles.groupby('frame').progress_apply(lambda x: self.particle_activity(x, pix2mm, xf_bed))

        print('Tracking particles...')
        if self.path.stem == 'edgertronic':
            pred = trackpy.predict.NearestVelocityPredict()
            tracks_bed = pred.link_df(particles[particles.activity==0], search_range=.5*self.d, adaptive_stop=self.d*.2, adaptive_step=0.95, memory=np.int(np.round(.1*self.fps)))
            tracks_bed_filtered = trackpy.filter_stubs(tracks_bed, np.round(.01*self.fps))

            # pred = trackpy.predict.NearestVelocityPredict()
            pred = trackpy.predict.ChannelPredict(3, 'x', minsamples=3)
            self.search_range = np.max([1.2*self.d, (2200 / (self.locations.pix_to_mm * self.info['frame_rate']))])
            # self.memory = np.int(np.round(.005*self.fps))
            self.memory = 5
            tracks_load = pred.link_df(particles[particles.activity==1], search_range=self.search_range, adaptive_stop=self.d/3, adaptive_step=0.95, memory=self.memory)
            print('For bed-load particle tracking, search range is %g particle diameters and memory is %i frames (Video frame rate is %g)' % (self.search_range/self.d, self.memory, self.info['frame_rate']))
            tracks_load_filtered = trackpy.filter_stubs(tracks_load, np.round(.005*self.fps))

            tracks_bed.particle += tracks_load.particle.max() + 1
            p_tracks = pd.concat([tracks_load, tracks_bed])
            # p_tracks = tracks_bed_filtered

        elif self.path.stem == 'manta':
            # pred = trackpy.predict.NearestVelocityPredict()
            search_range = (1200 / (self.locations.pix_to_mm * self.info['frame_rate']))
            bed_loc = int(self.bed.get().y_pix_bed.values.mean())
            initial_profile_guess = np.vstack([np.linspace(0,bed_loc,41), np.linspace(-search_range, 0, 41)]).swapaxes(0,1)
            pred = trackpy.predict.ChannelPredict(5, 'x', initial_profile_guess=initial_profile_guess, minsamples=5)
            p_tracks_load = pred.link_df(particles[particles.activity == 1], search_range=search_range, adaptive_stop=0.5, adaptive_step=0.98, memory=1)
            # p_tracks_load = trackpy.filtering.filter_stubs(p_tracks_load, threshold=5)

            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks_bed = pred.link_df(particles[particles.activity == 0], search_range=d/2, adaptive_stop=0.5, adaptive_step=0.98, memory=30)
            # p_tracks_bed = trackpy.filtering.filter_stubs(p_tracks_bed, threshold=5)

            p_tracks_bed.particle += p_tracks_load.particle.max() + 1
            p_tracks = pd.concat([p_tracks_load, p_tracks_bed])

        print('Smoothing tracks...')
        p_tracks = p_tracks.rename(index=str, columns={'x': 'x_pix', 'y': 'y_pix'})
        results = p_tracks.groupby('particle').progress_apply(self.sub_pxl_res).reset_index(drop=True)

        # save particle velocities
        if os.path.isfile(str(self.file_name)) is True:
            os.remove(str(self.file_name))

        xr.Dataset({
            'radius': ('ind', results.mass.values),
            'radius_sub': ('ind', results.radius_sub.values),
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


    def particle_activity(self, group, sf, xf_bed):
        bb = xf_bed.sel(frame=group.frame.values[0]).y_pix_bed.values #+ self.d/2
        x = group.x.values.astype(int)
        x[x >= 1200] = 1199
        dy = bb[x] - group.y.values
        group['dy_pix'] = dy
        group['activity'] = np.where(dy >= 0, 1, 0)
        return group


    def sub_pxl_res(self, group):
            group = group.reset_index(drop=True).set_index(np.array([np.float(x) for x in group.frame.values]))
            new_index = np.arange(np.int(group.frame.values[0]), np.int(group.frame.values[-1]))
            group = group.reindex(index=new_index).interpolate(method='linear', limit=205)
            group.frame = new_index

            activity = group.activity.mean()
            group['fractional_activity'] = activity

            N = group.frame.size
            if activity > 0.8:
                # n = int(np.round(0.05 * self.fps))
                n = 7
            else:
                # n = int(np.round(0.6 * self.fps))
                n = 101

            # if N < n:
            #     group['x_sub_pix'] = np.nan
            #     group['y_sub_pix'] = np.nan
            #     group['dy_sub_pix'] = np.nan
            #     group['radius_sub'] = np.nan
            # else:
            group['x_sub_pix'] = group.x_pix.rolling(window=n, win_type='hamming', center=True, min_periods=1).mean()
            group['y_sub_pix'] = group.y_pix.rolling(window=n, win_type='hamming', center=True, min_periods=1).mean()
            group['dy_sub_pix'] = group.dy_pix.rolling(window=n, win_type='hamming', center=True, min_periods=1).mean()
            group['radius_sub'] = group.mass.values.mean()
            return group
