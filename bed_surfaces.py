import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import pims
import h5py
import pandas as pd
import tqdm
import grain_locations
import grain_tracks
import rolling_mean
import pathlib
import xarray as xr

class bed_surfaces(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.info = vid_info
        self.pims_path = file_path
        self.path = file_path.parent

        if self.path.stem == 'manta':
            self.name = str(file_path.parent.parent.stem) + '_bed_surfaces.h5'
        elif self.path.stem == 'edgertronic':
            self.name = str(self.pims_path.stem) + '_bed_surfaces.h5'

        self.file_name = self.path / self.name
        self.locations = grain_locations.grain_locations(self.pims_path, vid_info)
        self.radius = self.locations.get_attr('mean_radius')

    # method to find already existing datasets
    def get(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        xf = xr.open_dataset(self.file_name)
        if frange[1] > xf.frame.max():
            return None
        else:
            return xf.where((xf.frame>=frange[0]) & (xf.frame<frange[1]), drop=True)

    def bed_line_preprocess(self, img):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # filter image
        img = cv.medianBlur(img, 7)

        # Apply thresholds
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, 1)

        # filter image
        img = cv.medianBlur(img, 5)

        return img

    def bed_calc(self, frange):
        # set number of frames to average over (odd numbers only)
        p = np.int(self.info['frame_rate'] / 2) # 1/2 a second
        p2 = np.int((np.floor(p/2) + 1))
        n_frames = np.int(frange[1] - frange[0])
        start_frame = np.int(frange[0] - p2) if np.int(frange[0] - p2) >= 0 else frange[0]
        # load all images (preprocessed for bed surface finding algorithm) into a numpy array
        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        imgs = np.zeros((n_frames, np.int(self.info['vertical_dim']), np.int(self.info['horizontal_dim'])))

        print('Loading images for bed surface function...')
        for frame in tqdm.tqdm(range(n_frames)):
            imgs[frame,:,:] = np.array(self.bed_line_preprocess(self.frames[frange[0] + frame]))

        print('Calculating bed surface...')

        # find bed surface
        frames = np.zeros(np.int(n_frames))
        bed_locs = np.zeros((np.int(n_frames), np.int(self.info['horizontal_dim'])))
        for frame in tqdm.tqdm(range(np.int(n_frames))):
            frame = frame
            if (n_frames - frame) < p2:
                # load images with adaptive threshold applied, already averaged over p images
                img = np.round(np.mean(imgs[-p:,:,:], axis=0)).astype(np.uint8)
            elif frame < p2:
                img = np.round(np.mean(imgs[:p,:,:], axis=0)).astype(np.uint8)
            else:
                img = np.round(np.mean(imgs[frame-p2+1:frame+p2,:,:], axis=0)).astype(np.uint8)


            # threshold the averaged images to get rid of ghost (moving) particles
            ret, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)

            # apply a morphological closing algorithm with a window slightly larger than a single bead. 2 time gets rid of all isolated beads in the bed
            # kernel = np.ones((self.rad*2, self.rad*2), np.uint8)
            X, Y = np.meshgrid(np.linspace(-self.radius/2, self.radius/2, self.radius+1), np.linspace(-self.radius/2, self.radius/2, self.radius+1))
            kernel = np.where(np.sqrt(X**2 + Y**2)<self.radius/2, 1, 0).astype(np.uint8)
            img = cv.morphologyEx(cv.bitwise_not(img), cv.MORPH_CLOSE, kernel, iterations=3)

            # reverse closing to get rid of noise above bed
            img = cv.morphologyEx(cv.bitwise_not(img), cv.MORPH_CLOSE, kernel, iterations=3)

            # apply an edge finding algorithm to the closed image
            img = cv.Canny(img,100,100)

            # find x and y coords of all bed line points
            y, x = np.where(img > 0)
            ind_sorted = x.argsort()
            x = x[ind_sorted]
            y = y[ind_sorted]
            y_out = np.zeros(np.int(self.info['horizontal_dim']))
            for ii in range(np.int(self.info['horizontal_dim'])):
                if not x[x==ii].size:
                    y_out[ii] = self.info['vertical_dim']
                else:
                    y_out[ii] = np.nanmean(y[x==ii])

            x = np.arange(np.int(self.info['horizontal_dim']))
            # y = y[:np.int(self.info['horizontal_dim'])]
            y_out = rolling_mean.rolling_mean(y_out, window_len=31)

            # save to dataframe
            frames[frame] = frange[0]+frame
            bed_locs[frame,:] = y_out

        # save particle velocities
        return xr.Dataset({
                'time': ('frame', frames*self.locations.dt),
                'x_pix_bed': (['frame', 'x'], np.array(frames.size*[np.arange(self.info['horizontal_dim'])])),
                'x_mm_bed': (['frame', 'x'], np.array(frames.size*[np.arange(self.info['horizontal_dim'])*self.locations.pix_to_mm])),
                'y_pix_bed': (['frame', 'x'], bed_locs),
                'y_mm_bed': (['frame', 'x'], bed_locs*self.locations.pix_to_mm)
                },
            coords={
                'frame': frames,
                'x': np.arange(self.info['horizontal_dim'])
            })

    def calculate(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        batch_size = 5000
        total_len = frange[1] - frange[0]
        batch = np.append(np.arange(0, total_len, batch_size), total_len)
        frange_list = [(batch[i], batch[i+1]) for i in range(batch.size-1)]
        xr.merge([self.bed_calc(range_in) for range_in in frange_list]).to_netcdf(self.file_name)

    def see_frame(self, frame_num, ret=None):

        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        img = self.frames[frame_num]
        xf = self.get(frange=(frame_num, frame_num+1))

        if xf is not None:
            frame = xf.frame.values[0]
            y = xf.y_pix_bed.values[0]
            # draw bed surface
            x = np.arange(y.size)
            pts_to_draw = np.array([[np.round(x[ii]*256).astype(np.int), np.round(y[ii]*256).astype(np.int)] for ii in range(x.size)])
            # draw bed line on image
            cv.polylines(img, np.int32([pts_to_draw]), False, (0, 0, 0), 2, cv.CV_AA, shift=8)

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
# obj = bed_surfaces(file_name)


# # method to overwrite an existing dataset with new data
# def overwrite_dataset(self, frange, data_in):
#     print('Saving bed surface...')
#
#     with h5py.File(self.file_name, 'r+') as f:
#         # set dataset name
#         dataset_name = 'frames_%ito%i' % frange
#
#         # check if group already exists
#         if not(self.group + '/' + dataset_name in f):
#
#             # create dataset using given name and data
#             f.create_dataset(self.group + '/' + dataset_name, data=data_in,
#                              maxshape=(None, None),
#                              fletcher32=True,
#                              shuffle=True,
#                              compression='lzf'
#                              )
#         else:
#             # overwrite specified dataset
#             f['/' + self.group + '/' + dataset_name].resize(data_in.shape)
#             f['/' + self.group + '/' + dataset_name][...] = data_in
#
#     # add attributes
#     self.make_dataset_attr(dataset_name, 'start_frame', frange[0])
#     self.make_dataset_attr(dataset_name, 'end_frame', frange[1])
#
# # method to add attributes to given dataset
# def make_dataset_attr(self, dataset_name, attribute_title, attribute_value):
#
#     # open file
#     with h5py.File(self.file_name, 'r+') as f:
#         grp = f['/' + self.group]
#         dset = grp[dataset_name]
#         dset.attrs[attribute_title] = attribute_value
#
# # method to get all attributes for a dataset
# def get_attrlist(self, dataset_name):
#
#     # open file
#     with h5py.File(self.file_name, 'r+') as f:
#         grp = f['/' + self.group]
#         dset = grp[dataset_name]
#         return [(name, val) for name, val in dset.attrs.items()]
#
# # method to get a specific attribute value for dataset
# def get_attr(self, dataset_name, attribute_name):
#
#     # open file
#     with h5py.File(self.file_name, 'r+') as f:
#         grp = f['/' + self.group]
#         dset = grp[dataset_name]
#         # ask for attribute, will return 'None' if the attribute does not exist
#         return dset.attrs.get(attribute_name)
