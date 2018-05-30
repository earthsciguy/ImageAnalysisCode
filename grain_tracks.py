import numpy as np
import pims
import os
import pandas as pd
import trackpy
import trackpy.predict as tp_predict
import grain_locations
import pathlib
import xarray as xr

class grain_tracks(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.info = vid_info
        self.pims_path = file_path
        self.path = file_path.parent
        self.name = str(file_path.parent.parent.stem) + '_tracks.h5'
        self.file_name = self.path / self.name
        self.locations = grain_locations.grain_locations(self.pims_path, vid_info)

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

        particles = rings.to_dataframe().reset_index(level=['frame']).rename(index=str, columns={'radius': 'mass', 'x_mm': 'x', 'y_mm': 'y'}).drop(columns=['x_pix', 'y_pix', 'time'])
        d = particles.mass.mean()*2

        print('Tracking particles...')
        pred = trackpy.predict.NearestVelocityPredict()
        p_tracks = pred.link_df(particles, search_range=2*d, adaptive_stop=0.5, adaptive_step=0.9, memory=3)

        # if self.name == 'edgertronic':
        #     # old search method updated sunday evening 28/1/2018
        #     # p_tracks = pred.link_df(particles, search_range=d, adaptive_stop=0.5, adaptive_step=0.99, memory=5)
        #     # new search method tried sunday evening 28/1/2018
        #     p_tracks = pred.link_df(particles, search_range=0.33*d, adaptive_stop=0.5, adaptive_step=0.9, memory=3)
        #
        # elif self.name == 'manta':
        #     pred = trackpy.predict.NearestVelocityPredict()
        #     p_tracks_near_bed = pred.link_df(particles_y_offset[(particles_y_offset.y_offset>=-2*d) & (particles_y_offset.y_offset<-d/2)], search_range=0.9*d, adaptive_stop=0.5, adaptive_step=0.98, memory=10)
        #
        #     pred = trackpy.predict.NearestVelocityPredict()
        #     p_tracks_bed = pred.link_df(particles_y_offset[(particles_y_offset.y_offset>=-d/2)], search_range=d/3, adaptive_stop=0.5, adaptive_step=0.98, memory=10)
        #
        #     p_tracks_near_bed.particle += p_tracks_load.particle.max() + 1
        #     p_tracks_bed.particle += p_tracks_near_bed.particle.max() + 1
        #     p_tracks = pd.concat([p_tracks_load, p_tracks_near_bed, p_tracks_bed])
        #
        # else:
        #     pred = trackpy.predict.NearestVelocityPredict()
        #     p_tracks = pred.link_df(particles, search_range=d, adaptive_stop=0.5, adaptive_step=0.99, memory=5)

        # save particle tracks
        self.tracks = xr.Dataset({
                'time': ('ind', rings.time.values),
                'radius': ('ind', rings.radius.values),
                'x_pix': ('ind', rings.x_pix.values),
                'x_mm': ('ind', rings.x_mm.values),
                'y_pix': ('ind', rings.y_pix.values),
                'y_mm': ('ind', rings.y_mm.values)
                },
            coords={
                'frame': ('ind', rings.frame.values),
                'particle': ('ind', p_tracks.particle.values)
                }).set_index(ind=['frame', 'particle'])

        if os.path.isfile(str(self.file_name)) is True: os.remove(str(self.file_name))
        self.tracks.reset_index('ind').to_netcdf(self.file_name)
