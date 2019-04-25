import pims
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path
from collections import Counter
import pandas as pd
import importlib
import re
import tqdm
tqdm.tqdm.pandas()
import grain_locations
importlib.reload(grain_locations)
import grain_tracks
importlib.reload(grain_tracks)
import grain_velocities
importlib.reload(grain_velocities)
import bed_surfaces
importlib.reload(bed_surfaces)
import area_weighted_quantities
importlib.reload(area_weighted_quantities)
import os


class manta_series(object):
    def __init__(self, file_path=None):

        self.name = 'manta'
        self.path = file_path
        self.movie_name = self.path.stem
        self.movie = pims.Video(str(self.path))
        def movie_info(video):
            return {
                    'frame_rate': 130,
                    'horizontal_dim': video.frame_shape[1],
                    'vertical_dim': video.frame_shape[0],
                    'duration': video._len / 130,
                    'frame_count': np.int(video._len)
                    }
        self.meta_data = movie_info(self.movie)
        # self.locs = grain_locations.grain_locations(self.path, vid_info=self.meta_data)
        # self.tracks = grain_tracks.grain_tracks(self.path, vid_info=self.meta_data)
        # self.vels = grain_velocities.grain_velocities(self.path, vid_info=self.meta_data)
        self.bed = bed_surfaces.bed_surfaces(self.path, vid_info=self.meta_data)

    def __repr__(self):
        s = ''
        for key, value in self.meta_data.items():
            s += '%s: %s\n' % (key, str(value))

        return s


class edgertronic_series(object):
    def __init__(self, file_path=None, exp_info=None):

        self.name = 'edgertronic'
        self.path = file_path
        self.movie_name = self.path.stem
        self.movie = pims.Video(str(self.path))
        def movie_info(path):
            df = pd.read_csv(path+'.txt', header=None, delimiter=':', nrows=26)
            return {
                    'start_time': df.iloc[0][1][1:] + ':' + df.iloc[0][2] + ':' + df.iloc[0][3][:-1],
                    'sensitivity': float(re.split(r'\t', df.iloc[2][1])[1]),
                    'shutter_speed': 1./float(re.split(r'\t', df.iloc[3][1])[1][2:]),
                    'frame_rate': float(re.split(r'\t', df.iloc[4][1])[1]),
                    'horizontal_dim': float(re.split(r'\t', df.iloc[5][1])[1]),
                    'vertical_dim': float(re.split(r'\t', df.iloc[6][1])[1]),
                    'duration': float(re.split(r'\t', df.iloc[8][1])[1]),
                    'frame_count': float(re.split(r'\t', df.iloc[25][1])[1])
                    }
        self.meta_data = movie_info(str(self.path).split('.mov')[0])
        self.locs = grain_locations.grain_locations(self.path, vid_info=self.meta_data)
        self.tracks = grain_tracks.grain_tracks(self.path, vid_info=self.meta_data)
        self.vels = grain_velocities.grain_velocities(self.path, vid_info=self.meta_data)
        self.bed = bed_surfaces.bed_surfaces(self.path, vid_info=self.meta_data)
        self.awq = area_weighted_quantities.area_weighted_quantities(self.path, vid_info=self.meta_data)

    def __repr__(self):
        s = ''
        for key, value in self.meta_data.items():
            s += '%s: %s\n' % (key, str(value))

        return s


class piv_series(object):
    def __init__(self, file_path=None, exp_info=None):

        self.name = 'piv'
        self.path = file_path / 'pims_images'
        # self.movie_name = self.path.stem
        self.movie = pims.ImageSequence(str(self.path / '*_0.tif'), process_func=lambda img: cv.convertScaleAbs(img, alpha=(255.0/4095.0)))
        def movie_info(video):
            return {
                    'frame_rate': 100,
                    'horizontal_dim': video.frame_shape[1],
                    'vertical_dim': video.frame_shape[0],
                    'duration': video._count / 100,
                    'frame_count': np.int(video._count)
                    }
        self.meta_data = movie_info(self.movie)
        # self.locs = grain_locations.grain_locations(self.path, vid_info=self.meta_data)
        # self.tracks = grain_tracks.grain_tracks(self.path, vid_info=self.meta_data)
        # self.vels = grain_velocities.grain_velocities(self.path, vid_info=self.meta_data)
        self.bed = bed_surfaces.bed_surfaces(self.path, vid_info=self.meta_data)
        # self.awq = area_weighted_quantities.area_weighted_quantities(self.path, vid_info=self.meta_data)

    def __repr__(self):
        s = ''
        for key, value in self.meta_data.items():
            s += '%s: %s\n' % (key, str(value))

        return s


class ldv_series(object):
    def __init__(self, file_path=None, exp_info=None):

        self.name = 'ldv'
        self.path = file_path[0].parent
        self.measurements = file_path

        y = [float([x[1:] for x in f.stem.split('d')[0].split('_') if 'Y' in x][0]) for f in self.measurements]
        vy_mean = [float(pd.read_csv(f, header=1)['Velocity Mean Ch. 1 (m/sec)'][0]) for f in self.measurements]
        vx_mean = [float(pd.read_csv(f, header=1)[' "Velocity Mean Ch. 2 (m/sec)"'][0]) for f in self.measurements]
        self.df = pd.DataFrame({'y': y, 'vx': vx_mean, 'vy': vy_mean}).sort_values('y')

    def plot(self):
        plt.semilogy(self.df.vx + self.df.vy, self.df.y - self.df.y.min() + 3, '-o', label=self.path.parent.stem)


class synoptic_series(object):
    def __init__(self, file_path=None, exp_info=None):

        def flip(img):
            rows,cols,ch = img.shape
            M = cv.getRotationMatrix2D((cols/2,rows/2),-2.5,1)
            img = cv.warpAffine(img,M,(cols,rows))
            return img[::-1,::-1,:][1200:-1020,:,:]

        self.name = 'synoptic'
        self.path = file_path.parent
        self.images = pims.ImageSequence(str(file_path) + '/*.JPG', process_func=flip)


class experiment(object):
    EXP_DATA = pd.read_csv(str(Path(os.path.dirname(__file__) + '/sed_trans_exp_obj_exp_data_all.txt')))
    # print(EXP_DATA.columns)

    # init method run when instance is created
    def __init__(self, file_path=None):

        # check if entered arguments are correct:
        if file_path is None:
            print('Please give file_path for experiment')

        else:
            self.path = file_path
            self._raw_info = experiment.EXP_DATA[experiment.EXP_DATA.names == str(self.path.stem)]
            self.name = self._raw_info.names.values[0]
            # print(self._raw_info.names)

            manta_path = file_path / 'manta'
#             if manta_path.exists():
#                 movies = [x for x in list(manta_path.glob('*.mp4')) if 'compressed' in str(x)]
#                 self.manta = [manta_series(file_path=x) for x in movies]
#             else:
            self.manta = []

            edgertronic_path = file_path / 'edgertronic'
            if edgertronic_path.exists():
                movies = [x for x in list(edgertronic_path.glob('*.mov')) if 'slomo' in str(x)]
                self.edgertronic = [edgertronic_series(file_path=x) for x in movies]
            else:
                self.edgertronic = []

            piv_path = file_path / 'piv'
#             if piv_path.exists():
#                 self.piv = piv_series(file_path=piv_path)
#             else:
            self.piv = []

            ldv_path = file_path / 'ldv'
            if ldv_path.exists():
                measurements = [x for x in list(ldv_path.glob('*.csv'))]
                self.ldv = ldv_series(measurements)
            else:
                self.ldv= []

            synoptic_path = file_path / 'synoptic'
            if synoptic_path.exists():
                self.synoptic = synoptic_series(synoptic_path)
            else:
                self.synoptic= []

            self.info = {
                'Experiment name': str(self.path.stem),
                'File path': str(self.path),
                'Grain kind': self._raw_info['grain_kind'].values[0],
                'Equilibrium mass flux (g/s)': self._raw_info['eq_mass_flux'].values[0]*1000.,
                'Nondimensional mass flux': self._raw_info['qs'].values[0],
                'Water discharge (l/s)': self._raw_info['discharge'].values[0],
                'Water surface slope (degrees)': self._raw_info['feed_mean_water_slope'].values[0],
                'Hydraulic radius (m)': self._raw_info['hyd_rad'].values[0],
                'Bed slope (degrees)': self._raw_info['feed_mean_bed_slope'].values[0],
                'Bed shear stress (Pa)': self._raw_info['taub'].values[0],
                'Nondimensional bed shear stress': self._raw_info['tau8'].values[0],
                'Edgertronic videos': len(self.edgertronic) if self.edgertronic else 0,
                'Edgertronic frames': int(sum([x.meta_data['frame_count'] for x in self.edgertronic])) if edgertronic_path else 0,
                'Manta videos': len(self.manta) if self.manta else 0,
                'Manta frames': int(sum([x.meta_data['frame_count'] for x in self.manta])) if manta_path else 0,
                # 'Canon images': self.canon.frames._count if self.canon.frames else 0,
                # 'Nikon image': self.nikon.frames._count if self.nikon.frames else 0,
                    }


    def __repr__(self):
        s = ''
        for key, value in self.info.items():
            s += '%s: %s\n' % (key, str(value))

        return s
