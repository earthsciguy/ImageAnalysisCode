import numpy as np
import pims
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
import ridge_directed_ring_detector as ring_detector
import h5py
import scipy.spatial.distance as dist
import pandas as pd
from multiprocessing import Pool, current_process


class grain_locations(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None):
        self.pims_path = file_path
        self.path = file_path.parent
        h5_name = str(file_path.stem) + '_grain_locations'
        self.file_name = file_path.parent / h5_name
        # self.make_group()
        self.info = vid_info


    # method to overwrite an existing dataset with new data
    def overwrite_dataset(self, frange, data_in):
        print 'Found %i particles. Saving...' % data_in.shape[0]

        file_name = self.file_name + '/locs_%i_%i.h5' % (frange[0], frange[1])
        with h5py.File(self.file_name, 'a') as f:
            # set dataset name
            dataset_name = 'frames_%ito%i' % frange

            # check if data already exists
            if not(dataset_name in f):

                # create dataset using given name and data
                f.create_dataset(dataset_name, data=data_in,
                                 maxshape=(None, None),
                                 fletcher32=True,
                                 shuffle=True,
                                 compression='lzf'
                                 )
            else:
                # overwrite specified dataset
                f['/' + dataset_name].resize(data_in.shape)
                f['/' + dataset_name][...] = data_in

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


    def batch_particle_locator(self, frange):
        print 'Processing frames %i to %i' % (frange[0], frange[1]-1)

        # find optimum gravel size
        self.w_ind = int(self.frames.frame_shape[1]/2.)
        ring_test = self.ridge_hough_rings(self.frames[0][:,self.w_ind:,:])
        self.rc = np.mean(ring_test[:,2])
        self.dr = np.std(ring_test[:,2])/2

        # define empty dataframe for found particle locations
        particles = pd.DataFrame()

        # iterate over each frame and save found particles
        for ii in tqdm.tqdm(np.arange(frange[0], frange[1])):
            # print('frame',ii)
            if self.path.stem is 'edgertronic':
                rings = self.ridge_hough_rings(self.frames[ii], [self.rc-self.dr, self.rc+self.dr])

            elif self.path.stem is 'manta':
                rings_1 = self.ridge_hough_rings(self.frames[ii][:,:self.w_ind,:], [self.rc-self.dr, self.rc+self.dr])
                rings_2 = self.ridge_hough_rings(self.frames[ii][:,self.w_ind:,:], [self.rc-self.dr, self.rc+self.dr])
                rings_2[:,1] += self.w_ind
                rings = np.vstack([rings_1, rings_2])

            for jj in range(rings.shape[0]):
                particles = particles.append([{
                                             'y': rings[jj,0],
                                             'x': rings[jj,1],
                                             'r': rings[jj,2],
                                             'frame': ii,
                                             },])

        self.find_datasets()
        self.overwrite_dataset((frange[0], frange[1]), particles))


    def get_locations(self, Trange=None, batch_size=100):
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

        franges = []
        for ii in np.arange(N_batches):
            franges.append((start_frames[ii], start_frames[ii+1]))

        p = Pool(3)
        p.map(self.batch_particle_locator, franges)


    def ridge_hough_rings(self, img, r=[10,100], ct=0.1):
        # if len(img.shape) > 2:
            # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        ridge_hough = ring_detector.RidgeHoughTransform(cv.bitwise_not(img).astype(np.float32))
        default_params = ridge_hough.params.copy()

        ridge_hough.params['circle_thresh'] = 2 * np.pi * ct
        ridge_hough.params['sigma'] = .05
        ridge_hough.params['curv_thresh'] = -100
        ridge_hough.params['Rmin'] = r[0]
        ridge_hough.params['Rmax'] = r[1]
        ridge_hough.params['vote_thresh'] = 7
        ridge_hough.params['dr'] = 2
        ridge_hough.params['eccentricity'] = 0
        # print 'starting preprocess'
        ridge_hough.img_preprocess()
        # print 'starting ridge hough'
        ridge_hough.rings_detection()
        # print 'finished ridge hough'
        ring = ridge_hough.output['rings_subpxl']
        # print 'finished outputting'
        dr = ring[:,2].mean()/3.
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
