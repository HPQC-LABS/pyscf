from mpi4py import MPI
import numpy as np
import h5py

comm = MPI.Comm.Get_parent()

# print(f'Hi from {comm.Get_rank()}/{comm.Get_size()}')

rank = comm.Get_rank()
n_procs = comm.Get_size()

row0 = None
row0 = comm.bcast(row0, root=0)

data_name = None
data_name = comm.bcast(data_name, root=0)

data_shape = None
data_shape = comm.bcast(data_shape, root=0)

test = np.empty(shape=data_shape)
comm.Bcast([test, MPI.DOUBLE], root=0)

data_size = data_shape[0]

if rank != n_procs - 1:
    chunk_size = data_size // n_procs
    start = rank * chunk_size
    stop = (rank + 1) * chunk_size
else:
    chunk_size = data_size % n_procs
    if chunk_size == 0: chunk_size = data_size // n_procs
    start = rank * (data_size // n_procs)
    stop = min((rank + 1) * (data_size // n_procs), data_size - 1) + 1

start += row0
stop += row0

print(f"rank {rank}: {start} - {stop}")

# comm.Barrier()
t1 = MPI.Wtime()

f = h5py.File('parallel_test.h5', 'a', driver='mpio', comm=MPI.COMM_WORLD)

if row0 < chunk_size:
    dset = f.create_dataset(data_name, data_shape, dtype='f8')
else:
    dset = f[data_name]
    
dset[start:stop] = test[start:stop]


t2 = MPI.Wtime()

# print(f"{rank}: {(t2 - t1)}")

f.close()
