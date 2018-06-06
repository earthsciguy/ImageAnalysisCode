import numpy as np
import pims
import os
import pandas as pd
import trackpy
from scipy import ndimage
import trackpy.predict as tp_predict
from dask.diagnostics import ProgressBar
import grain_locations
import pathlib
import xarray as xr

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

        self.sub_pxl_window = 5

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

        if self.path.stem == 'edgertronic':
            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks = pred.link_df(particles, search_range=1.5*d, adaptive_stop=0.5, adaptive_step=0.9, memory=5)

        #     # old search method updated sunday evening 28/1/2018
        #     # p_tracks = pred.link_df(particles, search_range=d, adaptive_stop=0.5, adaptive_step=0.99, memory=5)
        #     # new search method tried sunday evening 28/1/2018
        #     p_tracks = pred.link_df(particles, search_range=0.33*d, adaptive_stop=0.5, adaptive_step=0.9, memory=3)

        elif self.path.stem == 'manta':
            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks = pred.link_df(particles, search_range=d/2, adaptive_stop=0.5, adaptive_step=0.9, memory=5)

        #     pred = trackpy.predict.NearestVelocityPredict()
        #     p_tracks_near_bed = pred.link_df(particles_y_offset[(particles_y_offset.y_offset>=-2*d) & (particles_y_offset.y_offset<-d/2)], search_range=0.9*d, adaptive_stop=0.5, adaptive_step=0.98, memory=10)
        #
        #     pred = trackpy.predict.NearestVelocityPredict()
        #     p_tracks_bed = pred.link_df(particles_y_offset[(particles_y_offset.y_offset>=-d/2)], search_range=d/3, adaptive_stop=0.5, adaptive_step=0.98, memory=10)
        #
        #     p_tracks_near_bed.particle += p_tracks_load.particle.max() + 1
        #     p_tracks_bed.particle += p_tracks_near_bed.particle.max() + 1
        #     p_tracks = pd.concat([p_tracks_load, p_tracks_near_bed, p_tracks_bed])

        # save particle tracks
        df = xr.Dataset({
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
                }).set_index(ind=['frame', 'particle']).reset_index('particle').to_dask_dataframe().set_index('particle').rename(columns={'ind': 'frame'})

        for key in ['x_sub_pix', 'y_sub_pix']: df[key] = np.nan
        meta = {'frame': 'int64', 'time': 'float64', 'radius': 'float64', 'x_pix': 'float64', 'x_mm': 'float64', 'y_pix': 'float64', 'y_mm': 'float64', 'x_sub_pix': 'float64', 'y_sub_pix': 'float64'}
        delayed_obj = df.groupby(['particle']).apply(self.sub_pxl_res, meta=meta)

        print('Locating sub-pixel tracks...')
        with ProgressBar():
            results = delayed_obj.compute()

        # save particle velocities
        if os.path.isfile(str(self.file_name)) is True:
            os.remove(str(self.file_name))

        xr.Dataset({
            'time': ('ind', results.time.values),
            'radius': ('ind', results.radius.values),
            'x_pix': ('ind', results.x_pix.values),
            'x_mm': ('ind', results.x_mm.values),
            'y_pix': ('ind', results.y_pix.values),
            'y_mm': ('ind', results.y_mm.values),
            'x_sub_pix': ('ind', results.x_sub_pix.values),
            'x_sub_mm': ('ind', results.x_sub_pix.values*self.locations.pix_to_mm),
            'y_sub_pix': ('ind', results.y_sub_pix.values),
            'y_sub_mm': ('ind', results.y_sub_pix.values*self.locations.pix_to_mm),
            },
            coords={
                'frame': ('ind', results.frame.values),
                'particle': ('ind', results.index.values)
            }).to_netcdf(self.file_name)

    def sub_pxl_res(self, df):
        df.sort_values(by='frame', inplace=True)

        N = df.frame.size
        n = self.sub_pxl_window
        if N <= 2*n+1:
            return df

        dx = .5
        x_T = df.x_pix.values
        y_T = df.y_pix.values
        x_m = np.zeros(N)
        y_m = np.zeros(N)

        # handle end cases (make this fancier)
        x_m[:n] = x_T[:n]
        x_m[-n:] = x_T[-n:]
        y_m[:n] = y_T[:n]
        y_m[-n:] = y_T[-n:]

        # do middle
        for k in range(n,N-n+1):
            x = x_T[k-n:k+n]
            y = y_T[k-n:k+n]
            X, Y = np.meshgrid(np.arange(x.min(),x.max()+dx, dx), np.arange(y.min(), y.max()+dx, dx))
            loc = np.zeros_like(X)
            for i in range(2*n):
                loc += np.where(X == x[i], 1, 0) * np.where(Y == y[i], 1, 0)

            y_m_t, x_m_t = ndimage.measurements.center_of_mass(loc)
            x_m[k] = x.min()+x_m_t*dx
            y_m[k] = y.min()+y_m_t*dx

        # df['x_sub_pix'] = self.rolling_mean(x_m, window_len=n)
        df['x_sub_pix'] = x_m
        # df['y_sub_pix'] = self.rolling_mean(y_m, window_len=n)
        df['y_sub_pix'] = y_m

        return df

    def rolling_mean(self, x, window_len=11, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError#, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError#, "Input vector needs to be bigger than window size."


        if window_len<3:
            return x


        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError#, "Windowing function is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y[np.int((window_len-1)/2):-np.int((window_len-1)/2)]
