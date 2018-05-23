import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# import moviepy.editor as mpy
import multiprocessing
import pims
import pathlib
import h5py
import pandas as pd
import tqdm
import grain_locations
import grain_tracks
import xarray as xr
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class grain_velocities(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None, rotation=0):
        self.info = vid_info
        self.pims_path = file_path
        self.path = file_path.parent
        self.name = str(file_path.parent.parent.stem) + '_velocities.h5'
        self.file_name = self.path / self.name
        self.locations = grain_locations.grain_locations(self.pims_path, vid_info)
        self.tracks = grain_tracks.grain_tracks(self.pims_path, vid_info)
        self.theta = rotation # degrees to rotate bed
        self.min_track_life = 5

    # method to output dataset
    def get(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        xf = xr.open_dataset(self.file_name).set_index(ind=['frame', 'particle'])
        return xf.where((xf.frame>=frange[0]) & (xf.frame<frange[1]), drop=True)

    def velocity_calc(self, vels):
        print_info = (multiprocessing.current_process().name, vels.particle.values.min(), vels.particle.values.max())
        for particle, df_particle in vels.groupby('particle'):
            if df_particle.frame.values.size <= self.min_track_life:
                # print('deleted %i' % particle)
                vels = vels.where(vels.particle != particle, drop=True)

            else:
                # position of particle along track
                pos = {'x_mm': df_particle.x_mm.values, 'y_mm': df_particle.y_mm.values, 'x_pix': df_particle.x_pix.values, 'y_pix': df_particle.y_pix.values}

                # Calculate time difference between track locations
                dt = np.diff(df_particle.time.values)

                v = {}
                for key in pos:
                    # create empty vectors to be filled
                    v[key] = np.zeros_like(pos[key])

                    ################# Centered difference order 2
                    # # fill first and last elements
                    # v[key][0] = (pos[key][1] - pos[key][0]) / dt[0]
                    # v[key][-1] = (pos[key][-1] - pos[key][-2]) / dt[-1]
                    # # find velocity for middle elements
                    # for ii in range(1,dt.size):
                    #     v[key][ii] = (pos[key][ii+1] - pos[key][ii-1]) / (dt[ii-1:ii+1].sum())

                    ############### Centered difference order 4
                     # fill first and last elements
                    v[key][0] = (pos[key][1] - pos[key][0]) / dt[0]
                    v[key][1] = (pos[key][2] - pos[key][1]) / dt[1]
                    v[key][-2] = (pos[key][-2] - pos[key][-3]) / dt[-2]
                    v[key][-1] = (pos[key][-1] - pos[key][-2]) / dt[-1]
                    # find velocity for middle elements
                    for ii in range(2,dt.size-1):
                        v[key][ii] = (-(1/3)*pos[key][ii+2] + (8/3)*pos[key][ii+1] - (8/3)*pos[key][ii-1] + (1/3)*pos[key][ii-2]) / (dt[ii-2:ii+2].sum())

                    # save to original dataframe
                    # make a whole dataset for your new values
                    frames = vels.sel(particle=particle)['v'+key].frame.values
                    xfill = xr.Dataset(
                        {'v'+key: ('ind', v[key])},
                        coords={'frame': ('ind', frames), 'particle': ('ind', particle*np.ones(frames.size))}
                        ).set_index(ind=['frame', 'particle'])
                    # update values in original dataset
                    vels.merge(xfill, inplace=True)

        print('%s finished with frames %i to %i' % print_info)
        return vels

    def calculate(self, frange=None, batch_size=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        # open grain locations file
        self.vels = self.tracks.get(frange)

        print('Calculating grain velocities...')

        # prepare the new columns
        for key in ['vx_mm', 'vy_mm', 'vx_pix', 'vy_pix']:
            self.vels[key] = ('ind', np.nan*np.zeros_like(self.vels.radius.values))

        n = 4
        ranges = np.linspace(self.vels.particle.values.min(), self.vels.particle.values.max(), n+1).astype(int)
        args = [self.vels.where((self.vels.particle>=pr[0]) & (self.vels.particle<pr[1]), drop=True) for pr in [(ranges[i], ranges[i+1]) for i in range(n)]]
        p = multiprocessing.Pool(n)
        self.vels = xr.merge(list(p.map(self.velocity_calc, args)))

        # save particle velocities
        self.vels.reset_index('ind').to_netcdf(self.file_name)

    def see_frame(self, frame_num, ret=None):
        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        img = self.frames[frame_num]
        rings = self.get(frange=(frame_num, frame_num+1))
        s1 = 256*self.locations.pix_to_mm
        s2 = 140.
        if rings is not None:
            for row, ring in rings.groupby('particle'):
                # draw the center of the circle
                x = int(ring.x_pix.values*s1)
                y = int(ring.y_pix.values*s1)
                try:
                    vx = int(ring.x_pix.values*s1 + ring.vx_pix.values*s1/s2)
                    vy = int(ring.y_pix.values*s1 + ring.vy_pix.values*s1/s2)
                except:
                    continue
                rad = int(ring.radius.values*s1)
                color = (0,0,0)#((ring.particle % 10)*20., (ring.particle % 100)*2, (ring.particle % 1000)/4)
                # cv.arrowedLine(img, (x, y), (vx, vy), color, 2, cv.CV_AA, shift=8)
                cv.circle(img, (x, y), rad, color, 3, cv.LINE_AA, shift=8)
                # cv.circle(img, (x, y), 10, (0,0,0), 2, cv.LINE_AA, shift=8)

        if ret == 'image':
            return img
        else:
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    def make_movie(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        fps = 30
        num_images = frange[1] - frange[0]
        duration = num_images/fps

        def make_frame(t):
            return self.see_frame(frange[0] + int(t*fps), ret='image')

        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_videofile(str(self.file_name.parent / self.file_name.stem) + '.mp4', fps=fps) # export as video

# i = 1
# file_name = pathlib.Path('/Users/ericdeal/Dropbox (MIT)/3_postdoc/projects/sed_transport/1_data/0_main_feed_exp_data/_2017_exps/_data/glass_beads/exp_transport_stage_%i/manta/glass_beads_feed_n3p3_manta_record_130_playback_32.5.mp4'%i)
# obj = grain_velocities(file_name)
