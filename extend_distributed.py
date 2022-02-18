import os
import builtins
import numpy as np
import torch
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
torch_ccl = False
try:
    import torch_ccl
    torch_ccl = True
except ImportError as e:
    #print(e)
    torch_ccl = False
print(f"torch_ccl is {torch_ccl}")

my_rank = -1
my_size = -1
my_local_rank = -1
my_local_size = -1
alltoall_supported = False
allgatherv_supported = False
a2a_impl = os.environ.get('DLRM_ALLTOALL_IMPL', '')

def env2int(env_list, default = -1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default

def get_my_slice(n):
    my_size = dist.get_world_size()
    my_rank = dist.get_rank()
    k, m = divmod(n, my_size)
    return slice(my_rank * k + min(my_rank, m), (my_rank+1) * k + min(my_rank+1, m), 1)

def get_split_lengths(n):
    my_size = dist.get_world_size()
    k, m = divmod(n, my_size)
    if m == 0:
        splits = None
        my_len = k
    else:
        my_rank = dist.get_rank()
        splits = [(k+1) if i < m else k for i in range(my_size)]
        my_len = splits[my_rank]
    return (my_len, splits)

def init_distributed(rank = -1, size = -1, backend=''):
    global myreq
    global my_rank
    global my_size
    global my_local_rank
    global my_local_size
    global a2a_impl
    global alltoall_supported
    global allgatherv_supported

    # guess MPI ranks from env (works for IMPI, OMPI and MVAPICH2)
    num_mpi_ranks = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'])
    if backend == '' and num_mpi_ranks > 1:
        if torch_ccl and env2int(['CCL_WORKER_COUNT']) > 0:
            backend = 'ccl'
        elif dist.is_mpi_available():
            backend = 'mpi'
        else:
            print("WARNING: MPI multi-process launch detected but PyTorch MPI backend not available.")
            backend = 'gloo'
    if backend != '':
        #guess Rank and size
        if rank == -1:
            rank = env2int(['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK', 'RANK'], 0)
        if size == -1:
            size = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'], 1)
        if not os.environ.get('RANK', None) and rank != -1: os.environ['RANK'] = str(rank)
        if not os.environ.get('WORLD_SIZE', None) and size != -1: os.environ['WORLD_SIZE'] = str(size)
        if not os.environ.get('MASTER_PORT', None): os.environ['MASTER_PORT'] = '29500'
        if not os.environ.get('MASTER_ADDR', None):
            local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
            if local_size != size and backend != 'mpi':
                print("Warning: Looks like distributed multinode run but MASTER_ADDR env not set, using '127.0.0.1' as default")
                print("If this run hangs, try exporting rank 0's hostname as MASTER_ADDR")
            os.environ['MASTER_ADDR'] = '127.0.0.1'
    if size > 1:
        print(F"world_size:{size},rank:{rank}")
        dist.init_process_group(backend, rank=rank, world_size=size)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        my_local_rank = env2int(['MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
        my_local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
        if my_rank == 0: print("Running on %d ranks using %s backend" % (my_size, backend))
        if backend == 'ccl':
            print("Using CCL_ATL_TRANSPORT=%s" % os.environ.get('CCL_ATL_TRANSPORT', '(default)'))
            print("Using CCL_ATL_SHM=%s" % os.environ.get('CCL_ATL_SHM', '(default)'))
        if hasattr(dist, 'all_to_all_single'):
            try:
               # dist.all_to_all_single(torch.empty([0]), torch.empty([0]))
                alltoall_supported = True
            except RuntimeError:
                pass
        if a2a_impl == 'alltoall' and alltoall_supported == False:
            print("Requested DLRM_ALLTOALL_IMPL=%s but backend %s does not support it, use scatter/gather based alltoall" % (a2a_impl, backend))
            a2a_impl = 'scatter'
        if a2a_impl != '': print("Using DLRM_ALLTOALL_IMPL=%s" % a2a_impl)
        try:
            x = torch.ones([my_rank])
            y = torch.zeros([(my_size*(my_size-1))//2])
            y = list(y.split([r for r in range(my_size)]))
            dist.all_gather(y, x)
            allgatherv_supported = True
        except RuntimeError:
            pass
    else:
        my_rank = 0
        my_size = 1
        my_local_rank = 0
        my_local_size = 1

def barrier():
    if my_size > 1:
        dist.barrier()
