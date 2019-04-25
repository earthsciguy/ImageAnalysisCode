import numpy as np
import pims
import pandas as pd
import trackpy
from scipy import ndimage
import scipy.optimize as op
import trackpy.predict as tp_predict
import pathlib
import xarray as xr
import tqdm
import os
import grain_locations
import grain_tracks
import grain_velocities
import bed_surfaces
tqdm.tqdm.pandas(desc="Main loop")


class mass_flux(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.info = vid_info
        self.pims_path = file_path
        self.path = file_path.parent

        if self.path.stem == 'manta':
            self.name = str(file_path.parent.parent.stem) + '_mass_fluxes.h5'
        elif self.path.stem == 'edgertronic':
            self.name = str(self.pims_path.stem) + '_mass_fluxes.h5'

        self.file_name = self.path / self.name
        self.locs = grain_locations.grain_locations(self.pims_path, vid_info)
        self.tracks = grain_tracks.grain_tracks(self.pims_path, vid_info)
        self.bed = bed_surfaces.bed_surfaces(self.pims_path, vid_info)
        self.vels = grain_velocities.grain_velocities(self.pims_path, vid_info=vid_info)
        if os.path.isfile(str(self.vels.file_name)) is True:
            self.theta = self.theta_calc()
        else:
            self.theta = np.nan

        self.fps = vid_info['frame_rate']
        self.x_mark = int(vid_info['horizontal_dim']/2)

        xf = self.bed.get().groupby('x').mean()
        self.y_mark = (np.sin(self.theta) * xf.x.values + np.cos(self.theta) * xf.y_pix_bed_lin.values).max().astype(np.int) + 10


    # method to output dataset
    def get(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        xf = xr.open_dataset(self.file_name).set_index(ind=['frame'])
        return xf.where((xf.frame>=frange[0]) & (xf.frame<frange[1]), drop=True)


    def calculate(self, frange=None, batch_size=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        # Get particle velocities
        xf = self.vels.get(frange=frange)

        # Sort by particle number
        df = xf.reset_index('particle')
        df['frame'] = ('ind', xf.frame.values)

        # Filters out particles below the bed
        df = df.where(df.dy_sub_pix > 5, drop=True)

        ## Add rotated values
        df['x_sub_mm_rot'] = ('ind', np.cos(self.theta) * df.x_sub_mm.values - np.sin(self.theta) * df.y_sub_mm.values)
        df['x_sub_pix_rot'] = ('ind', np.cos(self.theta) * df.x_sub_pix.values - np.sin(self.theta) * df.y_sub_pix.values)
        df['y_sub_mm_rot'] = ('ind', np.sin(self.theta) * df.x_sub_mm.values + np.cos(self.theta) * df.y_sub_mm.values)
        df['vx_sub_mm_rot'] = ('ind', np.cos(self.theta) * df.vx_sub_mm.values - np.sin(self.theta) * df.vy_sub_mm.values)
        df['vy_sub_mm_rot'] = ('ind', np.sin(self.theta) * df.vx_sub_mm.values + np.cos(self.theta) * df.vy_sub_mm.values)
        df['x_mm_rot'] = ('ind', np.cos(self.theta) * df.x_mm.values - np.sin(self.theta) * df.y_mm.values)
        df['y_mm_rot'] = ('ind', np.sin(self.theta) * df.x_mm.values + np.cos(self.theta) * df.y_mm.values)
        df['vx_mm_rot'] = ('ind', np.cos(self.theta) * df.vx_mm.values - np.sin(self.theta) * df.vy_mm.values)
        df['vy_mm_rot'] = ('ind', np.sin(self.theta) * df.vx_mm.values + np.cos(self.theta) * df.vy_mm.values)

        ## Creates a new "particle crossings" column to add to each particle.
        df['particle_crossing'] = ('ind', np.zeros(df.particle.values.shape))

        ####################################
        ### Particle crossings per frame ###
        ####################################

        ## Populates the "particle crossings" column
        df = df.groupby('particle').apply(self.particle_activity).set_index(ind=['frame', 'particle']).reset_index('frame').to_dataframe().reset_index()

        df['qx_sub'] = np.zeros(df.frame.values.shape)
        df['qy_sub'] = np.zeros(df.frame.values.shape)
        df['qx'] = np.zeros(df.frame.values.shape)
        df['qy'] = np.zeros(df.frame.values.shape)

        ##########################
        ### q flux calculation ###
        ##########################
        df = df.groupby('frame').progress_apply(self.q_calc).groupby('frame').mean().reset_index()

        # save particle velocities
        if os.path.isfile(str(self.file_name)) is True:
            os.remove(str(self.file_name))

        xr.Dataset({
            'qx': ('frame', df.qx.values),
            'qy': ('frame', df.qy.values),
            'qx_sub': ('frame', df.qx_sub.values),
            'qy_sub': ('frame', df.qy_sub.values),
            'particle_crossing': ('frame', df.particle_crossing.values),
            },
            coords={
                'frame': ('frame', df.frame.values)
            }).to_netcdf(self.file_name)


    def theta_calc(self):
        vcs = self.vels.get().reset_index('ind').to_dataframe().reset_index()

        # Filter for above the bed:
        vcs = vcs[vcs.dy_sub_pix>5]   # 5 pixels above the bed.

        ## ANGLE FINDING:
        vxm = np.nanmean(vcs.vx_sub_pix.values)
        vym = np.nanmean(vcs.vy_sub_pix.values)

        def rot(phi):
            return abs(np.sin(phi)*vxm+np.cos(phi)*vym)

        return op.minimize(rot,0.0).x[0]


    def q_calc(self, group):
        intersecting_area_mm = np.pi * group.radius_mm.values**2 - (group.x_mm_rot.values - self.locs.pix_to_mm * self.x_mark)**2
        intersecting_area_sub_mm = np.pi * group.radius_sub_mm.values**2 - (group.x_sub_mm_rot.values - self.locs.pix_to_mm * self.x_mark)**2
        intersecting_area_mm[intersecting_area_mm < 0] = 0
        intersecting_area_sub_mm[intersecting_area_sub_mm < 0] = 0

        group['qx'] = -group.vx_mm * intersecting_area_mm / self.fps
        group['qx_sub'] = -group.vx_sub_mm * intersecting_area_sub_mm / self.fps

        intersecting_area_mm = np.pi * group.radius_mm.values**2 - (group.y_mm_rot.values - self.locs.pix_to_mm * self.y_mark)**2
        intersecting_area_sub_mm = np.pi * group.radius_sub_mm.values**2 - (group.y_sub_mm_rot.values - self.locs.pix_to_mm * self.y_mark)**2
        intersecting_area_mm[intersecting_area_mm < 0] = 0
        intersecting_area_sub_mm[intersecting_area_sub_mm < 0] = 0

        group['qy'] = -group.vy_mm * intersecting_area_mm / self.fps
        group['qy_sub'] = -group.vy_sub_mm * intersecting_area_sub_mm / self.fps
        return group


    ## Discrete particle counting (crossings per frame)
    def particle_activity(self, group):
        # Selects particles that do at some point cross the line
        if (group.x_sub_pix_rot.max() > self.x_mark) and (group.x_sub_pix_rot.min() < self.x_mark):
            # Once these particles are identified, the frame at which it crossed is marked with a 1.
            group.particle_crossing[np.argmin(np.abs(group.x_sub_pix_rot - self.x_mark))-1] += 1
        return group
