import numpy as np
import pims
import h5py
import pandas as pd
import xarray as xr
import os
import tqdm
tqdm.tqdm.pandas()
import grain_locations
import grain_tracks
import grain_velocities
import bed_surfaces

class area_weighted_quantities(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None, rotation=0):
        self.info = vid_info
        self.pims_path = file_path
        self.path = file_path.parent
        if self.path.stem == 'manta':
            self.name = str(file_path.parent.parent.stem) + '_area_weighted_quantities.h5'
        elif self.path.stem == 'edgertronic':
            self.name = str(self.pims_path.stem) + '_area_weighted_quantities.h5'

        self.file_name = self.path / self.name
        self.locs = grain_locations.grain_locations(self.pims_path, vid_info)
        self.tracks = grain_tracks.grain_tracks(self.pims_path, vid_info)
        self.bed = bed_surfaces.bed_surfaces(self.pims_path, vid_info)
        self.vels = grain_velocities.grain_velocities(self.pims_path, vid_info=vid_info)

        # choose bed slope
        self.theta = rotation # degrees to rotate bed
        self.dt = 1./self.info['frame_rate'] # time between frames


    # method to output dataset
    def get(self, frange=None, mean=False):
        if frange == None:
            frange = (0, self.info['frame_count'])

        xf = xr.open_dataset(self.file_name).set_index(ind=['frame', 'y_mm'])

        if mean is True:
            return xf.reset_index('ind').to_dataframe().reset_index().groupby('y_mm').mean().reset_index().drop(columns=['ind', 'frame'])
        else:
            return xf.where((xf.frame>=frange[0]) & (xf.frame<frange[1]), drop=True)


    def area_weighted_quantities_calc(self, df):
        Ay_t = np.zeros(self.pix_y_range.size)
        Vx_yt = np.zeros(self.pix_y_range.size)
        Vy_yt = np.zeros(self.pix_y_range.size)
        Phi_yt = np.zeros(self.pix_y_range.size)
        V_yt = np.zeros(self.pix_y_range.size)
        A_yo_corrected = np.zeros(self.pix_y_range.size)

        for ii, jj in enumerate(self.pix_y_range):
            df_j = df[df.radius_mm.values - np.abs(df.dy_sub_mm.values - jj) > 0]
            intersecting_radius = df_j.radius_mm.values**2 - (df_j.dy_sub_mm.values - jj)**2
            Ay_t[ii] = np.pi * np.nansum(intersecting_radius)
            A_yo_corrected[ii] = 10.1 * (df_j.x_sub_mm.max() - df_j.x_sub_mm.min() + 5)
            if jj < 0:
                Phi_yt[ii] = Ay_t[ii] / A_yo_corrected[ii]
            else:
                Phi_yt[ii] = Ay_t[ii] / self.A_yo

            Vx_yt[ii] = (np.pi * np.nansum(df_j.vx_sub_mm.values*(intersecting_radius))) / Ay_t[ii]
            Vy_yt[ii] = (np.pi * np.nansum(df_j.vy_sub_mm.values*(intersecting_radius))) / Ay_t[ii]
            V_yt[ii] = (np.pi * np.nansum(np.sqrt(df_j.vx_sub_mm.values**2 + df_j.vy_sub_mm.values**2)*(intersecting_radius))) / Ay_t[ii]

        df_out = pd.DataFrame({'A_yt': Ay_t, 'Vy_yt': Vy_yt, 'Vx_yt': Vx_yt, 'V_yt': V_yt, 'Phi_yt': Phi_yt}, index=self.pix_y_range)
        df_out.index.name = 'y'

        return df_out


    def calculate(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        print('Calculating area averaged velocities...')

        # open grain locations file
        df = self.vels.get(frange).reset_index('ind').to_dataframe().reset_index()
        self.pix_y_range = np.arange(np.int(df.dy_sub_pix.min()), np.int(df.dy_sub_pix.max()))
        self.A_yo = 10.1 * self.info['horizontal_dim'] * self.locs.pix_to_mm

        # calculate area_weighted_quantities
        results = df.groupby(['frame']).progress_apply(self.area_weighted_quantities_calc).reset_index()
        # results = results_by_frame.groupby(level=['y']).mean().reset_index()
        # results['y'] = results.y.values * self.locs.pix_to_mm

        # save particle velocities
        if os.path.isfile(str(self.file_name)) is True:
            os.remove(str(self.file_name))

        xf = xr.Dataset({
            'A_yt': ('ind', results.A_yt.values),
            'Vy_yt': ('ind', results.Vy_yt.values),
            'Vx_yt': ('ind', results.Vx_yt.values),
            'V_mag_yt': ('ind', results.V_yt.values),
            'Phi_yt': ('ind', results.Phi_yt.values),
            },
            coords={
                'frame': ('ind', results.frame.values),
                'y_mm': ('ind', results.y.values),
            }).to_netcdf(self.file_name)
