import numpy as np
import pims
import h5py
import pandas as pd
import xarray as xr
import os
import tqdm
tqdm.tqdm.pandas(desc="Main loop")
import grain_locations
import grain_tracks
import grain_velocities
import bed_surfaces

class area_weighted_quantities(object):

    # init method run when instance is created
    def __init__(self, locs, bed, tracks, vels, file_path=None, vid_info=None, rotation=0):
        self.info = vid_info
        self.pims_path = file_path
        self.path = file_path.parent
        if self.path.stem == 'manta':
            self.name = str(file_path.parent.parent.stem) + '_area_weighted_quantities.h5'
            self.profile_name = str(file_path.parent.parent.stem) + '_area_weighted_profiles.h5'
        elif self.path.stem == 'edgertronic':
            self.name = str(self.pims_path.stem) + '_area_weighted_quantities.h5'
            self.profile_name = str(self.pims_path.stem) + '_area_weighted_profiles.h5'

        self.file_name = self.path / self.name
        self.profile_file_name = self.path / self.profile_name
        # self.locs = grain_locations.grain_locations(self.pims_path, vid_info)
        # self.tracks = grain_tracks.grain_tracks(self.pims_path, vid_info)
        # self.bed = bed_surfaces.bed_surfaces(self.pims_path, vid_info)
        # self.vels = grain_velocities.grain_velocities(self.pims_path, vid_info=vid_info)
        self.locs = locs
        self.tracks = tracks
        self.bed = bed
        self.vels = vels

        # choose bed slope
        self.theta = rotation # degrees to rotate bed
        self.dt = 1./self.info['frame_rate'] # time between frames


    # method to output dataset
    def get_data(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        xf = xr.open_dataset(self.file_name).set_index(ind=['frame', 'y_mm'])
        return xf.where((xf.frame>=frange[0]) & (xf.frame<frange[1]), drop=True)


    # method to output profiles
    def get(self, frange=None):
        return xr.open_dataset(self.profile_file_name).set_index(ind=['y_mm'])


    def frame_area_weighted_quantities_calc(self, df):
        A_y = np.zeros(self.pix_y_range.size) # area of spheres intersecting plane at yi
        Phi_y = np.zeros(self.pix_y_range.size) # volume fraction of spheres at yi
        Vx_y = np.zeros(self.pix_y_range.size) # average x velocity of spheres intersecting plane weighted by intersection area
        Vy_y = np.zeros(self.pix_y_range.size) # average y velocity of spheres intersecting plane weighted by intersection area
        Vmag_y = np.zeros(self.pix_y_range.size) # magnitude of velocity of spheres intersecting plane weighted by intersection area

        for ii, yi in enumerate(tqdm.tqdm(self.pix_y_range, desc='mean field loop')):
            # find the subset of particles that intersect plane at yi
            df_j = df[df.radius_mm.values - np.abs(df.dy_sub_mm.values - yi) > 0]
            # radius circle of intersection of each sphere intersecting plane at yi
            # print(df_j.radius_mm.values.mean(), (df_j.dy_sub_mm.values - yi).mean())
            intersecting_radius_squared = df_j.radius_mm.values**2 - (df_j.dy_sub_mm.values - yi)**2
            # total area of intersecting spheres
            A_y[ii] = np.pi * np.nansum(intersecting_radius_squared)

            Vx_y[ii] = (np.pi * np.nansum(df_j.vx_sub_mm.values*intersecting_radius_squared)) / A_y[ii]
            Vy_y[ii] = (np.pi * np.nansum(df_j.vy_sub_mm.values*intersecting_radius_squared)) / A_y[ii]
            Vmag_y[ii] = (np.pi * np.nansum(np.sqrt(df_j.vx_sub_mm.values**2 + df_j.vy_sub_mm.values**2)*intersecting_radius_squared)) / A_y[ii]
            Phi_y[ii] = A_y[ii] / self.A_yo[ii]

        self.df_temp = pd.DataFrame({'Vx_y': Vx_y, 'Vy_y': Vy_y}, index=self.pix_y_range)
        self.df_temp.index.name = 'y'
        self.df_temp = self.df_temp.reset_index()

        # function to calculate deviation from mean velocity for each particles
        def dvi_calc(particle):
            for m in ['x', 'y']:
                particle['dv'+m+'_sub_mm'] = np.array([particle['v'+m+'_sub_mm'].values[ii] - self.df_temp['V'+m+'_y'][np.argmin(np.abs(self.df_temp.y - particle.y_mm.values[ii]))] for ii in range(particle.y_mm.values.size)])
            return particle

        # calculate deviations in velocities of particles relative to mean velocity Vm_y assessed at center of particle
        df['dvx_sub_mm'] = np.zeros_like(df.vx_pix.values) * np.nan
        df['dvy_sub_mm'] = np.zeros_like(df.vx_pix.values) * np.nan
        tqdm.tqdm.pandas(desc="deviatoric loop 1")
        df = df.groupby('particle').progress_apply(dvi_calc)

        # calculate mean deviatoric velocities
        A_y = np.zeros(self.pix_y_range.size) # area of spheres intersecting plane at yi
        Vxx_y = np.zeros(self.pix_y_range.size) # average x velocity of spheres intersecting plane weighted by intersection area
        Vyy_y = np.zeros(self.pix_y_range.size) # average y velocity of spheres intersecting plane weighted by intersection area
        Vxy_y = np.zeros(self.pix_y_range.size) # average y velocity of spheres intersecting plane weighted by intersection area

        for ii, yi in enumerate(tqdm.tqdm(self.pix_y_range, desc='deviatoric loop 2')):
            # find the subset of particles that intersect plane at yi
            df_j = df[df.radius_mm.values - np.abs(df.dy_sub_mm.values - yi) > 0]
            # radius circle of intersection of each sphere intersecting plane at yi
            intersecting_radius_squared = df_j.radius_mm.values**2 - (df_j.dy_sub_mm.values - yi)**2
            # total area of intersecting spheres
            A_y[ii] = np.pi * np.nansum(intersecting_radius_squared)

            Vxx_y[ii] = (np.pi * np.nansum(df_j.dvx_sub_mm.values*df_j.dvx_sub_mm.values*intersecting_radius_squared)) / A_y[ii]
            Vxy_y[ii] = (np.pi * np.nansum(df_j.dvx_sub_mm.values*df_j.dvy_sub_mm.values*intersecting_radius_squared)) / A_y[ii]
            Vyy_y[ii] = (np.pi * np.nansum(df_j.dvy_sub_mm.values*df_j.dvy_sub_mm.values*intersecting_radius_squared)) / A_y[ii]

        df_out = pd.DataFrame({'A_y': A_y, 'Vx_y': Vx_y, 'Vy_y': Vy_y, 'Vxx_y': Vxx_y, 'Vxy_y': Vxy_y, 'Vyy_y': Vyy_y, 'Vmag_y': Vmag_y, 'Phi_y': Phi_y}, index=self.pix_y_range)
        df_out.index.name = 'y'

        return df_out.reset_index()


    def movie_area_weighted_quantities_calc(self, df):
        A_y = np.zeros(self.pix_y_range.size) # area of spheres intersecting plane at yi
        Vx_y = np.zeros(self.pix_y_range.size) # average x velocity of spheres intersecting plane weighted by intersection area
        Vy_y = np.zeros(self.pix_y_range.size) # average y velocity of spheres intersecting plane weighted by intersection area
        Vmag_y = np.zeros(self.pix_y_range.size) # magnitude of velocity of spheres intersecting plane weighted by intersection area

        for ii, yi in enumerate(tqdm.tqdm(self.pix_y_range)):
            # find the subset of particles that intersect plane at yi
            df_j = df[df.radius_mm.values - np.abs(df.dy_sub_mm.values - yi) > 0]
            # radius circle of intersection of each sphere intersecting plane at yi
            # print(df_j.radius_mm.values.mean(), (df_j.dy_sub_mm.values - yi).mean())
            intersecting_radius_squared = df_j.radius_mm.values**2 - (df_j.dy_sub_mm.values - yi)**2
            # total area of intersecting spheres
            A_y[ii] = np.pi * np.nansum(intersecting_radius_squared)

            Vx_y[ii] = (np.pi * np.nansum(df_j.vx_sub_mm.values*intersecting_radius_squared)) / A_y[ii]
            Vy_y[ii] = (np.pi * np.nansum(df_j.vy_sub_mm.values*intersecting_radius_squared)) / A_y[ii]
            Vmag_y[ii] = (np.pi * np.nansum(np.sqrt(df_j.vx_sub_mm.values**2 + df_j.vy_sub_mm.values**2)*intersecting_radius_squared)) / A_y[ii]

        self.df_temp = pd.DataFrame({'Vx_y': Vx_y, 'Vy_y': Vy_y}, index=self.pix_y_range)
        self.df_temp.index.name = 'y'
        self.df_temp = self.df_temp.reset_index()

        # function to calculate deviation from mean velocity for each particles
        def dvi_calc(particle):
            for m in ['x', 'y']:
                particle['dv'+m+'_sub_mm'] = np.array([particle['v'+m+'_sub_mm'].values[ii] - self.df_temp['V'+m+'_y'][np.argmin(np.abs(self.df_temp.y - particle.y_mm.values[ii]))] for ii in range(particle.y_mm.values.size)])
            return particle

        # calculate deviations in velocities of particles relative to mean velocity Vm_y assessed at center of particle
        df['dvx_sub_mm'] = np.zeros_like(df.vx_pix.values) * np.nan
        df['dvy_sub_mm'] = np.zeros_like(df.vx_pix.values) * np.nan
        df = df.groupby('particle').progress_apply(dvi_calc)

        # calculate mean deviatoric velocities
        A_y = np.zeros(self.pix_y_range.size) # area of spheres intersecting plane at yi
        Vxx_y = np.zeros(self.pix_y_range.size) # average x velocity of spheres intersecting plane weighted by intersection area
        Vyy_y = np.zeros(self.pix_y_range.size) # average y velocity of spheres intersecting plane weighted by intersection area
        Vxy_y = np.zeros(self.pix_y_range.size) # average y velocity of spheres intersecting plane weighted by intersection area

        for ii, yi in enumerate(tqdm.tqdm(self.pix_y_range)):
            # find the subset of particles that intersect plane at yi
            df_j = df[df.radius_mm.values - np.abs(df.dy_sub_mm.values - yi) > 0]
            # radius circle of intersection of each sphere intersecting plane at yi
            intersecting_radius_squared = df_j.radius_mm.values**2 - (df_j.dy_sub_mm.values - yi)**2
            # total area of intersecting spheres
            A_y[ii] = np.pi * np.nansum(intersecting_radius_squared)

            Vxx_y[ii] = (np.pi * np.nansum(df_j.dvx_sub_mm.values*df_j.dvx_sub_mm.values*intersecting_radius_squared)) / A_y[ii]
            Vxy_y[ii] = (np.pi * np.nansum(df_j.dvx_sub_mm.values*df_j.dvy_sub_mm.values*intersecting_radius_squared)) / A_y[ii]
            Vyy_y[ii] = (np.pi * np.nansum(df_j.dvy_sub_mm.values*df_j.dvy_sub_mm.values*intersecting_radius_squared)) / A_y[ii]

        df_out = pd.DataFrame({'A_y': A_y, 'Vx_y': Vx_y, 'Vy_y': Vy_y, 'Vxx_y': Vxx_y, 'Vxy_y': Vxy_y, 'Vyy_y': Vyy_y, 'Vmag_y': Vmag_y}, index=self.pix_y_range)
        df_out.index.name = 'y'

        return df_out.reset_index()


    def calculate(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        print('Calculating area averaged velocities...')

        # open grain locations file
        df = self.vels.get(frange).reset_index('ind').to_dataframe().reset_index()
        self.pix_y_range = np.arange(np.int(df.dy_sub_pix.min()), np.int(df.dy_sub_pix.max()))
        W = self.info['horizontal_dim']
        H = self.info['vertical_dim']
        b = H-self.bed.get().y_pix_bed_lin.groupby('x').mean().values[0]
        m = -np.diff(self.bed.get().y_pix_bed_lin.groupby('x').mean().values).mean()
        width = np.zeros(self.pix_y_range.size)
        for i, y in enumerate(self.pix_y_range):
            if y < 0:
                width[i] = np.max([np.min([W + (b+y)/m, W]), 0])
            else:
                width[i] = np.min([(y+b-H)/-m, W])

        self.A_yo = 10.1 * width * self.locs.pix_to_mm

        # calculate area_weighted_quantities
        # results = self.area_weighted_quantities_calc(df.copy())
        # results = df.groupby(['frame']).progress_apply(self.frame_area_weighted_quantities_calc).reset_index()

        # # save frame-by-frame data
        # if os.path.isfile(str(self.file_name)) is True:
        #     os.remove(str(self.file_name))
        #
        # xf = xr.Dataset({
        #     'A_y': ('ind', results.A_y.values),
        #     'Vy_y': ('ind', results.Vy_y.values),
        #     'Vx_y': ('ind', results.Vx_y.values),
        #     'Vxx_y': ('ind', results.Vxx_y.values),
        #     'Vxy_y': ('ind', results.Vxy_y.values),
        #     'Vyy_y': ('ind', results.Vyy_y.values),
        #     'V_mag_yt': ('ind', results.Vmag_y.values),
        #     'Phi_y': ('ind', results.Phi_y.values),
        #     },
        #     coords={
        #         'frame': ('ind', results.frame.values),
        #         'y_mm': ('ind', results.y.values),
        #     }).to_netcdf(self.file_name)

        xf = self.get_data(frange)

        # average over frames and save as profile data
        mean_xf = xr.Dataset(xf.groupby('y_mm').mean().to_dataframe().rolling(11, center=True, win_type='cosine').mean()).rename({'Vxy_y':'Vxy_y_mean', 'Vx_y':'Vx_y_mean', 'Vy_y':'Vy_y_mean', 'Phi_y':'Phi_y_mean', 'Vyy_y':'Vyy_y_mean', 'Vxx_y':'Vxx_y_mean', 'V_mag_yt': 'V_mag_y_mean', 'A_y':'A_y_mean'})
        std_xf = xf.groupby('y_mm').std().rename({'Vxy_y':'Vxy_y_std', 'Vx_y':'Vx_y_std', 'Vy_y':'Vy_y_std', 'Phi_y':'Phi_y_std', 'Vyy_y':'Vyy_y_std', 'Vxx_y':'Vxx_y_std', 'V_mag_yt': 'V_mag_y_std', 'A_y':'A_y_std'})
        xr.merge([mean_xf, std_xf]).to_netcdf(self.profile_file_name)
