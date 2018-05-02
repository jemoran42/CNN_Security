import h5py

filename = 'test.h5'
c0_file = 'c0.h5'
c1_file = 'c1.h5'
c2_file = 'c2.h5'

hdf5_file = h5py.File(filename,'r')
c0_hdf5 = h5py.File(c0_file,'w')
c1_hdf5 = h5py.File(c1_file,'w')
c2_hdf5 = h5py.File(c2_file,'w')

data = hdf5_file['data']
label = hdf5_file['label']

c0_index = []
c1_index = []
c2_index = []

shape = label.shape
for i in range(0,shape[0]):
	if label[i,0] == 0.0:
		c0_index.append(i)
	elif label[i,0] == 1.0:
		c1_index.append(i)
	else:
		c2_index.append(i)


c0_hdf5.create_dataset("data", (len(c0_index),4), data.dtype)
c1_hdf5.create_dataset("data", (len(c1_index),4), data.dtype)
c2_hdf5.create_dataset("data", (len(c2_index),4), data.dtype)

c0_hdf5.create_dataset("label", (len(c0_index),1), label.dtype)
c1_hdf5.create_dataset("label", (len(c1_index),1), label.dtype)
c2_hdf5.create_dataset("label", (len(c2_index),1), label.dtype)

c0 = 0
c1 = 0
c2 = 0

for i in range(0,shape[0]):
	if i in c0_index:
		c0_hdf5["data"][c0,...] = hdf5_file["data"][i,...]
		c0_hdf5["label"][c0,0] = 0.0
		c0 = c0 + 1
	elif i in c1_index:
		c1_hdf5["data"][c1,...] = hdf5_file["data"][i,...]
		c1_hdf5["label"][c1,0] = 1.0
		c1 = c1 + 1
	else:
		c2_hdf5["data"][c2,...] = hdf5_file["data"][i,...]
		c2_hdf5["label"][c2,0] = 2.0
		c2 = c2 + 1

hdf5_file.close()
c0_hdf5.close()
c1_hdf5.close()
c2_hdf5.close()