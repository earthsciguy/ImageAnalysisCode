import numpy as np
import pims
import h5py
import pandas as pd
import trackpy
import trackpy.predict as tp_predict
import grain_locations


class grain_tracks(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.pims_path = file_path
        self.path = file_path.parent
        self.name = file_path.parent.stem
        h5_name_tracks = str(file_path.stem) + '_grain_tracks.h5'
        self.file_name = file_path.parent / h5_name_tracks

        self.locations = grain_locations.grain_locations(self.pims_path)
        self.make_group()
        self.info = vid_info


    def make_group(self):
        # open given hdf5 file, file is safely closed when with statement ends
        # print self.file_name
        with h5py.File(self.file_name, 'a') as f:

            if not '/grain_tracks' in f:
                grp = f.create_group('grain_tracks')
                self.group = grp.name
                self.dataset_names = []
                self.dataset_ranges = []

            else:
                # create a new link for the station instance
                grp = f['grain_tracks']
                # save group name for later use
                self.group = grp.name
                self.find_datasets()


    # method to overwrite an existing dataset with new data
    def overwrite_dataset(self, frange, data_in):
        print 'Found %i tracks. Saving...' % data_in.shape[0]

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


    def find_datasets(self):
        # open file
        with h5py.File(self.file_name, 'r+') as f:
            grp = f['/' + self.group]
            self.dataset_names = np.array([x for x in grp])
            self.dataset_ranges = np.array([[self.get_attrlist(x)[0][1], self.get_attrlist(x)[1][1]] for x in grp])


    # method to output dataset
    def get_frame_tracks(self, frange):
        self.find_datasets()
        if not np.array(self.dataset_ranges).any():
            return None

        else:
            dataset_names = self.dataset_names[np.where((frange[0] >= self.dataset_ranges[:,1]) | (frange[1] < self.dataset_ranges[:,0]), False, True)]

            # open given hdf5 file, file is safely closed when with statement ends
            with h5py.File(self.file_name, 'r+') as f:

                try:
                    # add group name to beginning of dataset name
                    dset = np.array([0, 0, 0, 0, 0])
                    for dataset_name in dataset_names:
                        path_name = self.group + '/' + dataset_name
                        dset = np.vstack([dset, f[path_name][...]])  # read dataset from hdf5 file

                    dset = dset[1:,:]
                    dset = dset[np.argsort(dset[:,0]),:]
                    if (dset[:,0][0] <= frange[0]) & (dset[:,0][-1] >= frange[1]-1):
                        return dset[(dset[:,0] >= frange[0]) & (dset[:,0] < frange[1]),:]
                    else:
                        return None

                except:
                    return None


    def batch_particle_tracker(self, frange):
        # open grain locations file
        rings = self.locations.get_frame_locations(frange)
        if rings is None:
            print 'Need to calculate grain locations...'
            return

        particles = pd.DataFrame({
            'frame':rings[:,0],
            'mass': rings[:,1],
            'x': rings[:,2],
            'y': rings[:,3]
                })
        d = particles.mass.mean()*2

        print 'Tracking particles...'

        if self.name == 'edgertronic':
            pred = trackpy.predict.NearestVelocityPredict()
            # old search method updated sunday evening 28/1/2018
            # p_tracks = pred.link_df(particles, search_range=d, adaptive_stop=0.5, adaptive_step=0.99, memory=5)
            # new search method tried sunday evening 28/1/2018
            p_tracks = pred.link_df(particles, search_range=0.33*d, adaptive_stop=0.5, adaptive_step=0.9, memory=3)

        elif self.name == 'manta':
            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks_near_bed = pred.link_df(particles_y_offset[(particles_y_offset.y_offset>=-2*d) & (particles_y_offset.y_offset<-d/2)], search_range=0.9*d, adaptive_stop=0.5, adaptive_step=0.98, memory=10)

            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks_bed = pred.link_df(particles_y_offset[(particles_y_offset.y_offset>=-d/2)], search_range=d/3, adaptive_stop=0.5, adaptive_step=0.98, memory=10)

            p_tracks_near_bed.particle += p_tracks_load.particle.max() + 1
            p_tracks_bed.particle += p_tracks_near_bed.particle.max() + 1
            p_tracks = pd.concat([p_tracks_load, p_tracks_near_bed, p_tracks_bed])

        else:
            pred = trackpy.predict.NearestVelocityPredict()
            p_tracks = pred.link_df(particles, search_range=d, adaptive_stop=0.5, adaptive_step=0.99, memory=5)

        # save particle tracks
        self.overwrite_dataset(frange, p_tracks)
        self.find_datasets()



    def get_tracks(self, Trange=None, batch_size=None):
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
            self.batch_particle_tracker((start_frames[batch], start_frames[batch+1]))
            # except:
                # fail[batch] = 1
                # pass

            self.find_datasets()
        return fail


    def see_frame(self, frame_num):

        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        img = self.frames[frame_num]

        tracks = self.get_frame_tracks(frange=(max(frame_num-200, 0), min(frame_num+200, self.info['frame_count'])))

        return tracks

        # if rings is not None:
        #     r = int(np.mean(rings[:,1]))
        #     for ring in rings:
        #         # draw the center of the circle
        #         cv.circle(img, (int(ring[2]),int(ring[3])), r, (0,0,0), 2, cv.CV_AA)
        #
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
