import numpy as np
import pims
import cv2 as cv
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
import ridge_directed_ring_detector as ring_detector
import h5py
import scipy.spatial.distance as dist
import pandas as pd
import tqdm
import matplotlib.pyplot as plt


class grain_locations(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.pims_path = file_path
        self.path = file_path.parent
        h5_name = str(file_path.stem) + '_grain_locations.h5'
        self.file_name = file_path.parent / h5_name
        self.make_group()
        self.info = vid_info


    def make_group(self):
        # open given hdf5 file, file is safely closed when with statement ends
        # print self.file_name
        with h5py.File(self.file_name, 'a') as f:

            if not '/grain_locations' in f:
                grp = f.create_group('grain_locations')
                self.group = grp.name
                self.dataset_names = []
                self.dataset_ranges = []

            else:
                # create a new link for the station instance
                grp = f['grain_locations']
                # save group name for later use
                self.group = grp.name
                self.find_datasets()


    # method to overwrite an existing dataset with new data
    def overwrite_dataset(self, frange, data_in):
        print 'Found %i particles. Saving...' % data_in.shape[0]

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
    def get_frame_locations(self, frange):
        self.find_datasets()
        if not np.array(self.dataset_ranges).any():
            return None

        else:
            dataset_names = self.dataset_names[np.where((frange[0] >= self.dataset_ranges[:,1]) | (frange[1] < self.dataset_ranges[:,0]), False, True)]

            # open given hdf5 file, file is safely closed when with statement ends
            with h5py.File(self.file_name, 'r+') as f:

                try:
                    # add group name to beginning of dataset name
                    dset = np.array([0, 0, 0, 0])
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


    def batch_particle_locator(self, frange):

        # prepare to load images through ridge_hough_rings function which finds particles as images are loaded

        # find optimum gravel size
        ring_test = self.ridge_hough_rings(self.frames[0])
        rc = np.mean(ring_test[:,2])
        dr = np.std(ring_test[:,2])/2

        # define empty dataframe for found particle locations
        particles = pd.DataFrame()

        # iterate over each frame and save found particles
        for ii in tqdm.tqdm(np.arange(frange[0], frange[1])):
            # print('frame',ii)
            rings = self.ridge_hough_rings(self.frames[ii], [rc-dr, rc+dr])
            for jj in range(rings.shape[0]):
                particles = particles.append([{
                                             'y': rings[jj,0],
                                             'x': rings[jj,1],
                                             'r': rings[jj,2],
                                             'frame': ii,
                                             },])

        self.overwrite_dataset(frange, particles)
        self.find_datasets()


    def get_locations(self, Trange=None, batch_size=1000):

        if Trange == None:
            Trange = (0, self.info['frame_count'])

        if (Trange[1] - Trange[0]) < batch_size:
            batch_size = Trange[1] - Trange[0] + 1

        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        # break into batches
        N_frames = Trange[1] - Trange[0]
        N_batches = np.floor((N_frames-1) / batch_size).astype(int) + 1
        start_frames = Trange[0] + batch_size * np.arange(N_batches+1)
        start_frames[-1] = Trange[1]

        fail = np.zeros(N_batches)
        for batch in range(N_batches):
            print 'Processing batch %i of %i' % (batch+1, N_batches)

            try:
                # locate particles
                self.batch_particle_locator((start_frames[batch], start_frames[batch+1]))
            except:
                fail[batch] = 1
                pass
        return fail


    def ridge_hough_rings(self, img, r=[10,100], ct=0.1):
        if len(img.shape) > 2:
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        ridge_hough = ring_detector.RidgeHoughTransform(cv.bitwise_not(img).astype(np.float32))
        default_params = ridge_hough.params.copy()

        ridge_hough.params['circle_thresh'] = 2 * np.pi * ct
        ridge_hough.params['sigma'] = .05
        ridge_hough.params['curv_thresh'] = -100
        ridge_hough.params['Rmin'] = r[0]
        ridge_hough.params['Rmax'] = r[1]
        ridge_hough.params['vote_thresh'] = 10
        ridge_hough.params['dr'] = 2
        ridge_hough.params['eccentricity'] = 0
        # print 'starting preprocess'
        ridge_hough.img_preprocess()
        # print 'starting ridge hough'
        ridge_hough.rings_detection()
        # print 'finished ridge hough'
        ring = ridge_hough.output['rings_subpxl']
        # print 'finished outputting'
        dr = ring[:,2].mean()/10.
        # find distances between all particles
        distances = dist.cdist(ring[:,:2], ring[:,:2])
        # find particles that are closer than dr and are not themselves
        x, y = np.where(distances<dr)
        d_ring = x[x!=y]
        # filter out those close particles and take average of their location
        close_rings = ring[d_ring, :]
        # print close_rings.size
        new_rings = []
        while close_rings.size > 0:
            close_ps = dist.cdist(close_rings[0,:2].reshape((1,2)), close_rings[:,:2])
            xp, yp = np.where(close_ps<dr)
            new_rings.append(np.mean(close_rings[yp,:], axis=0))
            close_rings = close_rings[np.delete(np.arange(close_rings.shape[0]), yp)]
        new_rings = np.array(new_rings)
        # delete double particles from ring array, and replace them with averaged locations
        ind = np.arange(ring.shape[0])
        ind = np.delete(ind, d_ring)
        ring = ring[ind,:]
        # print(ring.shape)
        if new_rings.size > 0:
            ring = np.vstack([ring, new_rings])

        return ring


    def see_frame(self, frame_num):

        if (self.pims_path.suffix == '.mov') or (self.pims_path.suffix == '.mp4'):
            # Load videos
            self.frames = pims.Video(str(self.pims_path))
        else:
            self.frames = pims.ImageSequence(str(self.pims_path))

        img = self.frames[frame_num]

        rings = self.get_frame_locations(frange=(frame_num, frame_num+1))

        if rings is not None:
            r = int(np.mean(rings[:,1]))
            for ring in rings:
                # draw the center of the circle
                cv.circle(img, (int(ring[2]),int(ring[3])), r, (0,0,0), 2, cv.CV_AA)

        plt.imshow(img)
        plt.axis('off')
        plt.show()
