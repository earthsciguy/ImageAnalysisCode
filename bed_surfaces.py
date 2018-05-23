import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# import moviepy.editor as mpy
import pims
import h5py
import pandas as pd
import tqdm
import grain_locations
import grain_tracks
import rolling_mean


class bed_surfaces(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.pims_path = file_path
        self.path = file_path.parent
        self.name = file_path.parent.stem
        h5_name_tracks = str(file_path.stem) + '_bed_surface.h5'
        self.file_name = file_path.parent / h5_name_tracks

        self.info = vid_info
        # try:
            # self.rad = self.locations.get_frame_locations((0,1))[:,1].mean()
        # except:
        self.rad = 21

        self.make_group()


    def make_group(self):
        # open given hdf5 file, file is safely closed when with statement ends
        # print self.file_name
        with h5py.File(self.file_name, 'a') as f:

            if not '/bed_surface' in f:
                grp = f.create_group('bed_surface')
                self.group = grp.name
                self.dataset_names = []
                self.dataset_ranges = []

            else:
                # create a new link for the station instance
                grp = f['bed_surface']
                # save group name for later use
                self.group = grp.name
                self.find_datasets()


    # method to overwrite an existing dataset with new data
    def overwrite_dataset(self, frange, data_in):
        print('Saving bed surface...')

        with h5py.File(self.file_name, 'r+') as f:
            # set dataset name
            dataset_name = 'frames_%ito%i' % frange

            # check if group already exists
            if not(self.group + '/' + dataset_name in f):

                # create dataset using given name and data
                f.create_dataset(self.group + '/' + dataset_name, data=data_in,
                                 maxshape=(None, None),
                                 fletcher32=True,
                                 shuffle=True,
                                 compression='lzf'
                                 )
            else:
                # overwrite specified dataset
                f['/' + self.group + '/' + dataset_name].resize(data_in.shape)
                f['/' + self.group + '/' + dataset_name][...] = data_in

        # add attributes
        self.make_dataset_attr(dataset_name, 'start_frame', frange[0])
        self.make_dataset_attr(dataset_name, 'end_frame', frange[1])


    # method to add attributes to given dataset
    def make_dataset_attr(self, dataset_name, attribute_title, attribute_value):

        # open file
        with h5py.File(self.file_name, 'r+') as f:
            grp = f['/' + self.group]
            dset = grp[dataset_name]
            dset.attrs[attribute_title] = attribute_value


    # method to get all attributes for a dataset
    def get_attrlist(self, dataset_name):

        # open file
        with h5py.File(self.file_name, 'r+') as f:
            grp = f['/' + self.group]
            dset = grp[dataset_name]
            return [(name, val) for name, val in dset.attrs.items()]


    # method to get a specific attribute value for dataset
    def get_attr(self, dataset_name, attribute_name):

        # open file
        with h5py.File(self.file_name, 'r+') as f:
            grp = f['/' + self.group]
            dset = grp[dataset_name]
            # ask for attribute, will return 'None' if the attribute does not exist
            return dset.attrs.get(attribute_name)


    # method to find already existing datasets
    def find_datasets(self):
        # open file
        with h5py.File(self.file_name, 'r+') as f:
            grp = f['/' + self.group]
            self.dataset_names = np.array([x for x in grp])
            self.dataset_ranges = np.array([[self.get_attrlist(x)[0][1], self.get_attrlist(x)[1][1]] for x in grp])


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

 ####################################### must modify below this line ##################


    def bed_line_preprocess(self, img):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # filter image
        img = cv.medianBlur(img, 7)

        # Apply thresholds
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, 1)

        # filter image
        img = cv.medianBlur(img, 5)

        return img


    def get_bed_surfaces(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        # set number of frames to average over (odd numbers only)
        p = np.int(self.info['frame_rate'] / 1.5) # 1/2 a second
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
        df_bed_surface = np.zeros((np.int(n_frames), np.int(self.info['horizontal_dim'])+1))
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
            X, Y = np.meshgrid(np.linspace(-self.rad/2, self.rad/2, self.rad+1), np.linspace(-self.rad/2, self.rad/2, self.rad+1))
            kernel = np.where(np.sqrt(X**2 + Y**2)<self.rad/2, 1, 0).astype(np.uint8)
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
            y_out = rolling_mean.rolling_mean(y_out, window_len=34)

            # save to dataframe
            df_bed_surface[frame, 0] = frange[0]+frame
            df_bed_surface[frame, 1:] = y_out

        # save particle velocities
        self.overwrite_dataset(frange, df_bed_surface)
        self.find_datasets()


    # method to output dataset
    def get_frame_bed_surfaces(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        dataset_names = self.dataset_names[np.where((frange[0] >= self.dataset_ranges[:,1]) | (frange[1] < self.dataset_ranges[:,0]), False, True)]

        # open given hdf5 file, file is safely closed when with statement ends
        with h5py.File(self.file_name, 'r+') as f:

            try:
                # add group name to beginning of dataset name
                dset = np.zeros((np.int(self.info['horizontal_dim'])+1))
                for dataset_name in dataset_names:
                    path_name = self.group + '/' + dataset_name
                    dset = np.vstack([dset, f[path_name][...]])  # read dataset from hdf5 file

                dset = dset[1:,:]
                dset = dset[np.argsort(dset[:,0]),:]
                if (dset[0,:][0] <= frange[0]) & (dset[:,0][-1] >= frange[1]-1):
                    bed_data = dset[(dset[:,0] >= frange[0]) & (dset[:,0] < frange[1]),:]
                    return bed_data[:,0].astype(int), bed_data[:,1:].squeeze()

                else:
                    return None

            except:
                return None


    def see_frame(self, frame_num, ret=None):

        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        img = self.frames[frame_num]
        frame, y = self.get_frame_bed_surfaces(frange=(frame_num, frame_num+1))

        if y is not None:
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
