import mpi4py
from mpi4py import MPI
import numpy as np
import logging


TAG_CNT_PACKETS = 11
TAG_PAYLOAD = 12

mpi4py.rc.threaded = False

class MPILogger:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.comm = MPI.COMM_WORLD

    def debug(self, msg):
        # if self.comm.Get_rank() == 0:
        # self.logger.debug(msg)
        print(msg)


logger = MPILogger()



def get_cnt_samples_per_worker(size_data, cnt_shares):
    return (size_data + cnt_shares - 1) // cnt_shares


def get_ids_per_receiver(
        id_worker, cnt_samples_per_worker, cnt_workers, size_data, pad):
    assert id_worker < cnt_workers
    ids = []
    for i in range(cnt_samples_per_worker):
        next_id = (id_worker + i * cnt_workers)
        if not pad and (next_id >= size_data):
            break
        next_id %= size_data
        ids.append(next_id)
    ids = sorted(ids)
    return np.array(ids)


def get_local_subindices(indices, source_lo, source_hi):
    lo = np.searchsorted(indices, source_lo)
    hi = np.searchsorted(indices, source_hi - 1, side='right')
    return lo, hi


class DataSource:

    def __init__(self, data, comm):
        self.comm = comm
        self.data = data
        self.size_local = len(data)
        self.size_global = comm.allreduce(self.size_local)
        self.lo, self.hi = self.get_local_range()
        # print(f"ds on rank {comm.Get_rank()}, lo={self.lo}, hi={self.hi}")

    def get_local_range(self):
        hi = self.comm.scan(self.size_local)
        lo = hi - self.size_local
        return lo, hi


def shuffle(src, dst, comm, csize, rank, pad=False):
    status = MPI.Status()
    ranks = comm.allgather(comm.Get_rank())
    data_source = DataSource(src, comm)

    recv_index = ranks.index(rank) + 1
    if recv_index == csize: recv_index = 0
    send_index = ranks.index(rank) - 1
    if send_index == -1: send_index = csize - 1
    cnt_samples_per_worker = get_cnt_samples_per_worker(data_source.size_global, csize)

    for step in range(2 * (csize - 1)):
        # even start sending
        if step % 2 == rank % 2:
            recver = ranks[recv_index]

            # prepare data
            #ids = get_ids_per_receiver(recver, cnt_samples_per_worker, csize, data_source.size_global, pad)
            #lo, hi = get_local_subindices(ids, data_source.lo, data_source.hi)
            #send_buf = []
            #if lo < hi:
                #send_buf = [data_source.data[i - data_source.lo] for i in ids[lo:hi]]

            # comm.send(len(send_buf), dest=recver, tag=TAG_CNT_PACKETS)
            comm.send(rank, dest=recver, tag=TAG_CNT_PACKETS)
            print(f"step {step}: sender {rank} sent to {recver}")
            #if len(send_buf):
                #comm.send(send_buf, dest=recver, tag=TAG_PAYLOAD)

            recv_index += 1
            if recv_index == csize: recv_index = 0

        # odd start receiving
        else:
            sender = ranks[send_index]

            getsize = comm.recv(source=sender, tag=TAG_CNT_PACKETS, status=status)
            print(f"step {step}: recver {rank} got from {getsize}")
            #if getsize:
                #buf = comm.recv(source=sender, tag=TAG_PAYLOAD, status=status)
                #dst += buf

            send_index -= 1
            if send_index == -1: send_index = csize - 1

    print(f"sender {comm.Get_rank()}   done")


def main():
    comm = MPI.COMM_WORLD
    csize = comm.Get_size()
    rank = comm.Get_rank()

    comm.Barrier()
    if rank == 0:
        print("############################# STAGE 2 ##############################")
    logging.basicConfig(level=logging.DEBUG)
    local_data = []
    if rank % 8 == 0:
        local_data = [np.random.random((100, 100)) for i in range(1000)]
    comm.Barrier()

    received_payload = []
    shuffle(local_data, received_payload, comm, csize, rank, pad=True)
    comm.Barrier()
    print(f"rank {rank}   received  {len(received_payload)}")
    comm.Barrier()
    if rank == 0:
        print(f"done!")


if __name__ == "__main__":
    main()
