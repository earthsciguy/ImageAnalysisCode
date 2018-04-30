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


class grain_velocities(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None, rotation=0):
        self.pims_path = file_path
        self.path = file_path.parent
        self.name = file_path.parent.stem
        h5_name_tracks = str(file_path.stem) + '_grain_velocities.h5'
        self.file_name = file_path.parent / h5_name_tracks

        self.locations = grain_locations.grain_locations(self.pims_path)
        self.tracks = grain_tracks.grain_tracks(self.pims_path)

        # choose bed slope
        self.theta = rotation # degrees to rotate bed
        self.info = vid_info
        self.dt = 1./self.info['frame_rate'] # time between frames
        # self.pixel_per_mm = self.info['pixel_d']
        try:
            self.pixel_per_mm = self.locations.get_frame_locations((0,1))[:,1].mean()/2.5
        except:
            return

        self.make_group()


    def make_group(self):
        # open given hdf5 file, file is safely closed when with statement ends
        # print self.file_name
        with h5py.File(self.file_name, 'a') as f:

            if not '/grain_velocities' in f:
                grp = f.create_group('grain_velocities')
                self.group = grp.name
                self.dataset_names = []
                self.dataset_ranges = []

            else:
                # create a new link for the station instance
                grp = f['grain_velocities']
                # save group name for later use
                self.group = grp.name
                self.find_datasets()


    # method to overwrite an existing dataset with new data
    def overwrite_dataset(self, frange, data_in):
        print 'Saving velocities...'

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
            return [(name, val) for name, val in dset.attrs.iteritems()]


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


    # method to output dataset
    def get_frame_grain_velocities(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        dataset_names = self.dataset_names[np.where((frange[0] >= self.dataset_ranges[:,1]) | (frange[1] < self.dataset_ranges[:,0]), False, True)]

        # open given hdf5 file, file is safely closed when with statement ends
        with h5py.File(self.file_name, 'r+') as f:

            try:
                # add group name to beginning of dataset name
                dset = np.array([0, 0, 0, 0, 0, 0, 0])
                for dataset_name in dataset_names:
                    path_name = self.group + '/' + dataset_name
                    dset = np.vstack([dset, f[path_name][...]])  # read dataset from hdf5 file

                dset = dset[1:,:]
                dset = dset[np.argsort(dset[:,0]),:]
                if (dset[:,0][0] <= frange[0]) & (dset[:,0][-1] >= frange[1]-1):
                    velocity_data = dset[(dset[:,0] >= frange[0]) & (dset[:,0] < frange[1]),:]
                    df_velocities = pd.DataFrame({
                                        'frame': velocity_data[:,0].astype(int),
                                        'rad': velocity_data[:,2],
                                        # rotate coordinates by theta degrees
                                        'x': velocity_data[:,5],
                                        'vx': velocity_data[:,3],
                                        'y': velocity_data[:,6],
                                        'vy': velocity_data[:,4],
                                        'particle': velocity_data[:,1].astype(int),
                                                })
                    return df_velocities

                else:
                    return None

            except:
                return None


    def batch_grain_velocities(self, frange):
        # open grain locations file
        tracks = self.tracks.get_frame_tracks(frange)
        if tracks is None:
            print 'Need to calculate grain tracks...'
            return

        # convert distance to millimeters
        df_tracks = pd.DataFrame({
                            'frame': tracks[:,0].astype(int),
                            'rad': tracks[:,1]/self.pixel_per_mm,
                            # rotate coordinates by theta degrees
                            'x': (tracks[:,2]*np.cos(np.radians(self.theta)) - tracks[:,3]*np.sin(np.radians(self.theta)))/self.pixel_per_mm,
                            'vx': np.zeros(tracks.shape[0])/self.pixel_per_mm,
                            'y': (tracks[:,2]*np.sin(np.radians(self.theta)) + tracks[:,3]*np.cos(np.radians(self.theta)))/self.pixel_per_mm,
                            'vy': np.zeros(tracks.shape[0])/self.pixel_per_mm,
                            'particle': tracks[:,4].astype(int),
                                    })

        print 'Calculating grain velocities...'

        for particle in tqdm.tqdm(range(df_tracks.particle.values.max())):
            # get particle track
            df_particle = df_tracks[df_tracks.particle == particle]
            if df_particle.shape[0] < 10:
                df_tracks = df_tracks.drop(df_tracks.loc[df_tracks.particle==particle].index)

            else:
                # x position of particle along track
                x = df_particle.x.values
                # y position of particle along track
                y = df_particle.y.values
                # Calculate time difference between track locations
                dt = np.diff(df_particle.frame.values)*self.dt

                # create empty vectors to be filled
                vx = np.zeros_like(x)
                vy = np.zeros_like(x)

                ################# Centered difference order 2
                # # fill first and last elements
                # vx[0] = (x[1] - x[0]) / dt[0]
                # vx[-1] = (x[-1] - x[-2]) / dt[-1]
                # vy[0] = (y[1] - y[0]) / dt[0]
                # vy[-1] = (y[-1] - y[-2]) / dt[-1]
                # # find velocity for middle elements
                # for ii in range(1,dt.size):
                #     vx[ii] = (x[ii+1] - x[ii-1]) / (dt[ii-1:ii+1].sum())
                #     vy[ii] = (y[ii+1] - y[ii-1]) / (dt[ii-1:ii+1].sum())


                ############### Centered difference order 4
                 # fill first and last elements
                vx[0] = (x[1] - x[0]) / dt[0]
                vx[1] = (x[2] - x[1]) / dt[1]
                vx[-2] = (x[-2] - x[-3]) / dt[-2]
                vx[-1] = (x[-1] - x[-2]) / dt[-1]
                vy[0] = (y[1] - y[0]) / dt[0]
                vy[1] = (y[2] - y[1]) / dt[1]
                vy[-2] = (y[-2] - y[-3]) / dt[-2]
                vy[-1] = (y[-1] - y[-2]) / dt[-1]
                # find velocity for middle elements
                for ii in range(2,dt.size-1):
                    vx[ii] = (-(1/3)*x[ii+2] + (8/3)*x[ii+1] - (8/3)*x[ii-1] + (1/3)*x[ii-2]) / (dt[ii-2:ii+2].sum())
                    vy[ii] = (-(1/3)*y[ii+2] + (8/3)*y[ii+1] - (8/3)*y[ii-1] + (1/3)*y[ii-2]) / (dt[ii-2:ii+2].sum())


                # save to original dataframe
                df_tracks.loc[df_tracks.particle==particle, 'vx'] = vx
                df_tracks.loc[df_tracks.particle==particle, 'vy'] = vy

        # save particle velocities
        self.overwrite_dataset(frange, df_tracks)
        self.find_datasets()


    def get_velocities(self, Trange=None, batch_size=None):
        if Trange == None:
            Trange = (0, self.info['frame_count'])

        if batch_size == None:
            batch_size = Trange[1] - Trange[0] + 1

        # break into batches
        N_frames = Trange[1] - Trange[0]
        N_batches = np.floor((N_frames-1) / batch_size).astype(int) + 1
        start_frames = Trange[0] + batch_size * np.arange(N_batches+1)
        start_frames[-1] = Trange[1]

        fail = np.zeros(N_batches)
        for batch in range(N_batches):
            print 'Processing batch %i of %i' % (batch+1, N_batches)

            # try:
                # locate particles
            self.batch_grain_velocities((start_frames[batch], start_frames[batch+1]))
            # except:
                # fail[batch] = 1
                # pass

        return fail


    def see_frame(self, frame_num, ret=None):

        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        img = self.frames[frame_num]
        rings = self.get_frame_grain_velocities(frange=(frame_num, frame_num+1))
        s1 = 256*self.pixel_per_mm
        s2 = 140.
        if rings is not None:
            for row, ring in rings.iterrows():
                # draw the center of the circle
                x = int(ring.x*s1)
                y = int(ring.y*s1)
                try:
                    vx = int(ring.x*s1 + ring.vx*s1/s2)
                    vy = int(ring.y*s1 + ring.vy*s1/s2)
                except:
                    continue
                rad = int(ring.rad*s1)
                color = (0,0,0)#((ring.particle % 10)*20., (ring.particle % 100)*2, (ring.particle % 1000)/4)
                # cv.arrowedLine(img, (x, y), (vx, vy), color, 2, cv.CV_AA, shift=8)
                cv.circle(img, (x, y), rad, color, 3, cv.CV_AA, shift=8)
                # cv.circle(img, (x, y), 10, (0,0,0), 2, cv.CV_AA, shift=8)

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
