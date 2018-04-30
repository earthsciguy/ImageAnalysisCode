import numpy as np
import pims
import h5py
import pandas as pd
import tqdm
import grain_locations
import grain_tracks
import grain_velocities


class area_weighted_quantities(object):

    # init method run when instance is created
    def __init__(self, file_path=None, vid_info=None, rotation=0):
        self.pims_path = file_path
        self.path = file_path.parent
        self.name = file_path.parent.stem
        h5_name_tracks = str(file_path.stem) + '_area_weighted_quantities.h5'
        self.file_name = file_path.parent / h5_name_tracks

        self.locations = grain_locations.grain_locations(self.pims_path)
        self.tracks = grain_tracks.grain_tracks(self.pims_path)
        self.velocities = grain_velocities.grain_velocities(self.pims_path, vid_info=vid_info)

        # choose bed slope
        self.theta = rotation # degrees to rotate bed
        self.info = vid_info
        self.dt = 1./self.info['frame_rate'] # time between frames
        # self.pixel_d = self.info['pixel_d']
        # self.pixel_d = 2.5 # pixels per mm

        self.make_group()


    def make_group(self):
        # open given hdf5 file, file is safely closed when with statement ends
        # print self.file_name
        with h5py.File(self.file_name, 'a') as f:

            if not '/area_weighted_quantities' in f:
                grp = f.create_group('area_weighted_quantities')
                self.group = grp.name
                self.dataset_names = []
                self.dataset_ranges = []

            else:
                # create a new link for the station instance
                grp = f['area_weighted_quantities']
                # save group name for later use
                self.group = grp.name
                self.find_datasets()


    # method to overwrite an existing dataset with new data
    def overwrite_dataset(self, frange, data_name, data_in):
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
    def get_frame_area_weighted_quantities(self, frange=None):
        if frange == None:
            frange = (0, self.info['frame_count'])

        dataset_names = self.dataset_names[np.where((frange[0] >= self.dataset_ranges[:,1]) | (frange[1] < self.dataset_ranges[:,0]), False, True)]

        # open given hdf5 file, file is safely closed when with statement ends
        with h5py.File(self.file_name, 'r+') as f:

            # try:
            # add group name to beginning of dataset name
            dset = np.array([])
            for dataset_name in dataset_names:
                path_name = self.group + '/' + dataset_name
                return np.nanmean(f[path_name][...], axis=0)
                # dset = np.vstack([dset, f[path_name][...]])  # read dataset from hdf5 file

            #     dset = dset[1:,:]
            #     dset = dset[np.argsort(dset[:,0]),:]
            #     if (dset[:,0][0] <= frange[0]) & (dset[:,0][-1] >= frange[1]-1):
            #         velocity_data = dset[(dset[:,0] >= frange[0]) & (dset[:,0] < frange[1]),:]
            #         df_velocities = pd.DataFrame({
            #                             'frame': velocity_data[:,0].astype(int),
            #                             'rad': velocity_data[:,2],
            #                             # rotate coordinates by theta degrees
            #                             'x': velocity_data[:,5],
            #                             'vx': velocity_data[:,3],
            #                             'y': velocity_data[:,6],
            #                             'vy': velocity_data[:,4],
            #                             'particle': velocity_data[:,1].astype(int),
            #                                     })
            #         return df_velocities
            #
            #     else:
            #         return None
            #
            # except:
            #     return None


    def batch_area_weighted_quantities(self, frange, nj=100):
        df_vels = self.velocities.get_frame_grain_velocities(frange)

        # calculate Ay and Ax
        n_frames = df_vels.frame.max() - df_vels.frame.min()

        # xtotal = df_vels.x.values.max() - df_vels.x.values.min() + 5
        # Ayo = 10.1 * xtotal
        # ytotal = df_vels.y.values.max() - df_vels.y.values.min() + 5
        # Axo = 10.1 * ytotal

        # nk = 500
        # xk = np.linspace(df_vels.x.values.min(), df_vels.x.values.max(), nk)
        # Axt = np.zeros((n_frames, nk))
        # Phi_xt = np.zeros((n_frames, nk))
        # Vy_xt = np.zeros((n_frames, nk))
        # Vx_xt = np.zeros((n_frames, nk))
        # V2_xt = np.zeros((n_frames, nk))

        yj = np.linspace(50, 0, nj)
        Ayt = np.zeros((n_frames, nj))
        Phi_yt = np.zeros((n_frames, nj))
        Vx_yt = np.zeros((n_frames, nj))
        Vy_yt = np.zeros((n_frames, nj))
        V2_yt = np.zeros((n_frames, nj))
        print 'Finding area averaged quantities of particles...'
        for ff in tqdm.tqdm(range(n_frames)):
            df_frame = df_vels[df_vels.frame==ff]
            # for kk in range(nk):
            #     df_k = df_frame[df_frame.rad.values - np.abs(df_frame.x.values - xk[kk]) > 0]
            #     Axt[ff, kk] = np.pi * np.sum(df_k.rad.values**2 - (df_k.x.values - xk[kk])**2)
            #     Phi_xt[ff, kk] = Axt[ff, kk] / Axo
            #     Vx_xt[ff, kk] = (np.pi * np.sum( df_k.vx.values*(df_k.rad.values**2 - (df_k.x.values - xk[kk])**2) )) / Axt[ff, kk]
            #     Vy_xt[ff, kk] = (np.pi * np.sum( df_k.vy.values*(df_k.rad.values**2 - (df_k.x.values - xk[kk])**2) )) / Axt[ff, kk]
            #     V2_xt[ff, kk] = (np.pi * np.sum( (df_k.vx.values**2 + df_k.vy.values**2)*(df_k.rad.values**2 - (df_k.x.values - xk[kk])**2) )) / Axt[ff, kk]

            for jj in range(nj):
                df_j = df_frame[df_frame.rad.values - np.abs(df_frame.y.values - yj[jj]) > 0]
                temp = df_j.rad.values**2 - (df_j.y.values - yj[jj])**2
                Ayt[ff, jj] = np.pi * np.sum(temp)
                # Phi_yt[ff, jj] = Ayt[ff, jj] / Ayo
                Vx_yt[ff, jj] = (np.pi * np.sum(df_j.vx.values*(temp))) / Ayt[ff, jj]
                Vy_yt[ff, jj] = (np.pi * np.sum(df_j.vy.values*(temp))) / Ayt[ff, jj]
                # V2_yt[ff, jj] = (np.pi * np.sum((df_j.vx.values**2 + df_j.vy.values**2)*(temp))) / Ayt[ff, jj]

        datas = {
            # 'Axt': Axt, 'Phi_xt': Phi_xt, 'Vx_xt': Vx_xt, 'Vy_xt': Vy_xt, 'V2_xt': V2_xt,
            'Ayt': Ayt, 'Phi_yt': Phi_yt, 'Vx_yt': Vx_yt, 'Vy_yt': Vy_yt, 'V2_yt': V2_yt
                }

        # save particle velocities
        self.overwrite_dataset(frange, 'Vx_yt', Vx_yt)
        self.find_datasets()
        # for name, data in datas.iteritems():
        #     overwrite_dataset(exp+'_'+name+'.hdf5', 'frames_%ito%i' % (start_frames[batch], end_frames[batch]), name, data)


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
            self.batch_area_weighted_quantities((start_frames[batch], start_frames[batch+1]))
            # except:
                # fail[batch] = 1
                # pass

        return fail
