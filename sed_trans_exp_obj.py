import pims
import numpy as np
# from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path
from collections import Counter
import pandas as pd
import importlib
import os
import re
import grain_locations
importlib.reload(grain_locations)
import grain_tracks
importlib.reload(grain_tracks)
import grain_velocities
importlib.reload(grain_velocities)
import bed_surfaces
importlib.reload(bed_surfaces)
# import area_weighted_quantities
# importlib.reload(area_weighted_quantities)


class image_series(object):
    def __init__(self, camera_name=None, file_path=None):

        self.name = camera_name
        self.path = file_path / self.name
        if self.path.exists():
            counter = 0
            for (file_ext,count) in Counter([x.suffix for x in self.path.iterdir()]).items():
                if count > counter:
                    counter = count
                    img_ext = file_ext

            self.frames = pims.ImageSequence(str(self.path.absolute())+'/*'+img_ext)

            self.locations = grain_locations.grain_locations(self.path/self.name)
            self.tracks = grain_tracks.grain_tracks(self.path/self.name)
            # self.grain_velocities = grain_velocities.grain_velocities(self.path/self.name)
            # self.info {
            #                 'frame_rate': 130,
            #                 'horizontal_dim': video.frame_shape[1],
            #                 'vertical_dim': video.frame_shape[0],
            #                 'duration': video._len/130.,
            #                 'frame_count': video._len
            #                 }

        if not self.path.exists():
            self.path = None
            self.frames = None
            self.locations = None
            self.tracks = None
            self.grain_velocities = None


    def __repr__(self):
        return str(self.frames)


class manta_series(object):
    def __init__(self, file_path=None):

        self.name = 'manta'
        self.path = file_path / self.name
        if self.path.exists():
            self.movie_names = [x.stem for x in self.path.glob('*.mp4')]
            self.movie_paths = [x for x in self.path.glob('*.mp4')]
            self.movies_ = dict(zip(self.movie_names, [pims.Video(str(x)) for x in self.movie_paths]))
            self.movies = [self.movies_[x] for x in self.movie_names]

            def movie_info(video):
                return {
                        'frame_rate': 130,
                        'horizontal_dim': video.frame_shape[1],
                        'vertical_dim': video.frame_shape[0],
                        'duration': video._len / 130,
                        'frame_count': np.int(video._len)
                        }
            self.meta_data_ = dict(zip(self.movie_names, [movie_info(x) for x in self.movies]))
            self.meta_data = [self.meta_data_[x] for x in self.movie_names]
            self.locations_ = dict(zip(self.movie_names, [grain_locations.grain_locations(x, vid_info=self.meta_data_[x.stem]) for x in self.movie_paths]))
            self.locations = [self.locations_[x] for x in self.movie_names]
            self.tracks_ = dict(zip(self.movie_names, [grain_tracks.grain_tracks(x, vid_info=self.meta_data_[x.stem]) for x in self.movie_paths]))
            self.tracks = [self.tracks_[x] for x in self.movie_names]
            self.vels_ = dict(zip(self.movie_names, [grain_velocities.grain_velocities(x, vid_info=self.meta_data_[x.stem]) for x in self.movie_paths]))
            self.vels = [self.vels_[x] for x in self.movie_names]
            # self.bed_surfaces_ = dict(zip(self.movie_names, [bed_surfaces.bed_surfaces(x, vid_info=self.meta_data_[x.stem]) for x in self.movie_paths]))
            # self.bed_surfaces = [self.bed_surfaces[x] for x in self.movie_names]

        if not self.path.exists():
            self.path = None
            self.movie_names = None
            self.movie_paths = None
            self.meta_paths = None
            self.movies = None
            self.meta_data = None
            self.movies_ = None
            self.meta_data_ = None
            self.locations = None
            self.locations_ = None
            self.tracks = None
            self.tracks_ = None
            self.vels = None
            self.vels_ = None
            self.bed_surfaces = None
            self.bed_surfaces_ = None


class edgertronic_series(object):
    def __init__(self, file_path=None, exp_info=None):

        self.name = 'edgertronic'
        self.path = file_path / self.name
        if self.path.exists():
            self.movie_names = [x.stem for x in self.path.glob('*.mov')]
            self.movie_paths = [x for x in self.path.glob('*.mov')]
            self.meta_paths = [x for x in self.path.glob('*.txt')]
            self.movies_ = dict(zip(self.movie_names, [pims.Video(str(x)) for x in self.movie_paths]))
            self.movies = [self.movies_[x] for x in self.movie_names]

            def movie_info(path):
                df = pd.read_csv(str(path), header=None, delimiter=':', nrows=26)
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

            self.meta_data_ = dict(zip(self.movie_names, [movie_info(x) for x in self.meta_paths]))
            self.meta_data = [self.meta_data_[x] for x in self.movie_names]
            self.locations_ = dict(zip(self.movie_names, [grain_locations.grain_locations(x, vid_info=self.meta_data_[x.stem]) for x in self.movie_paths]))
            self.locations = [self.locations_[x] for x in self.movie_names]
            self.tracks_ = dict(zip(self.movie_names, [grain_tracks.grain_tracks(x, vid_info=self.meta_data_[x.stem]) for x in self.movie_paths]))
            self.tracks = [self.tracks_[x] for x in self.movie_names]
            self.vels_ = dict(zip(self.movie_names, [grain_velocities.grain_velocities(x, vid_info=self.meta_data_[x.stem]) for x in self.movie_paths]))
            self.vels = [self.vels_[x] for x in self.movie_names]
            # self.area_weighted_quantities = dict(zip(self.movie_names, [area_weighted_quantities.area_weighted_quantities(x, vid_info=self.meta_data[x.stem]) for x in self.movie_paths]))
            # self.area_weighted_quantities_ = [self.area_weighted_quantities[x] for x in self.movie_names]
            # self.bed_surfaces = dict(zip(self.movie_names, [bed_surfaces.bed_surfaces(x, vid_info=self.meta_data_[x.stem]) for x in self.movie_paths]))
            # self.bed_surfaces_ = [self.bed_surfaces[x] for x in self.movie_names]


        if not self.path.exists():
            self.path = None
            self.movie_names = None
            self.movie_paths = None
            self.meta_paths = None
            self.movies = None
            self.meta_data = None
            self.movies_ = None
            self.meta_data_ = None
            self.locations = None
            self.locations_ = None
            self.tracks = None
            self.tracks_ = None
            self.grain_velocities = None
            self.grain_velocities_ = None
            self.area_weighted_quantities = None
            self.area_weighted_quantities_ = None
            self.bed_surfaces = None
            self.bed_surfaces_ = None


class experiment(object):
    EXP_DATA = pd.read_csv(str(Path(os.path.dirname(__file__) + '/sed_trans_exp_obj_exp_data_GB.txt')))

    # init method run when instance is created
    def __init__(self, file_path=None):

        # check if entered arguments are correct:
        if file_path is None:
            print('Please give file_path for experiment')

        else:
            self.path = file_path
            self._raw_info = experiment.EXP_DATA[experiment.EXP_DATA.names == str(self.path.stem)]

            self.edgertronic = edgertronic_series(file_path=self.path)
            self.manta = manta_series(file_path=self.path)

            # self.piv = image_series(camera_name='piv', file_path=self.path)
            # self.canon = image_series(camera_name='bed_TL', file_path=self.path)
            # self.nikon = image_series(camera_name='synoptic_TL', file_path=self.path)

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
                'Edgertronic videos': len(self.edgertronic.movie_paths) if self.edgertronic.movie_paths else 0,
                'Edgertronic frames': int(sum([y['frame_count'] for x, y in self.edgertronic.meta_data_.items()])) if self.edgertronic.movie_paths else 0,
                'Manta videos': len(self.manta.movie_paths) if self.manta.movie_paths else 0,
                'Manta frames': int(sum([y['frame_count'] for x, y in self.manta.meta_data_.items()])) if self.manta.movie_paths else 0,
                # 'Canon images': self.canon.frames._count if self.canon.frames else 0,
                # 'Nikon image': self.nikon.frames._count if self.nikon.frames else 0,
                    }


    def __repr__(self):
        s = ''
        for key, value in self.info.items():
            s += '%s: %s\n' % (key, str(value))

        return s

# legend = {
# 'feed_n3p3': 'exp_transport_stage_1',
# 'feed_n1p5': 'exp_transport_stage_2',
# 'feed_0p5': 'exp_transport_stage_3',
# 'feed_1p3': 'exp_transport_stage_4',
# 'feed_2p2_1': 'exp_transport_stage_5',
# 'feed_2p2_2': 'exp_transport_stage_6',
# 'feed_5p2_1': 'exp_transport_stage_7',
# 'feed_5p2_2': 'exp_transport_stage_8',
# 'feed_12p95': 'exp_transport_stage_9',
# 'feed_13p0': 'exp_transport_stage_10',
# 'feed_20p0': 'exp_transport_stage_11',
# 'feed_22p5': 'exp_transport_stage_12'
# }
# x = pd.read_pickle('sed_trans_exp_obj_exp_data.pd')
# x = x[x.grain_kind == '5mm_glass']
# x['names'] = [legend[name] for name in list(x.index.values)]
# x = x.set_index('names')
# x.to_csv('sed_trans_exp_obj_exp_data_GB.txt')
