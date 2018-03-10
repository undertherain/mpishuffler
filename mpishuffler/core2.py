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

    def get_data_for_receiver(self, id_receiver, cnt_receivers, pad):
        cnt_samples_per_worker = get_cnt_samples_per_worker(self.size_global, cnt_receivers)
        ids = get_ids_per_receiver(id_receiver, cnt_samples_per_worker, cnt_receivers, self.size_global, pad)
        lo, hi = get_local_subindices(ids, self.lo, self.hi)
        if lo < hi:
            send_buf = [self.data[i - self.lo] for i in ids[lo:hi]]
        else:
            send_buf = []
        return send_buf


def shuffle(src, dst, comm, pad=False, count_me_in=True):
    csize = comm.Get_size()
    rank = comm.Get_rank()
    ranks = comm.allgather(comm.Get_rank())
    ranks_receivers = comm.allgather(rank if count_me_in else -1)
    #ranks_receivers = [i for i in ranks_receivers if i >= 0]
    status = MPI.Status()
    data_source = DataSource(src, comm)

    toRight = ranks.index(rank) + 1
    if toRight == csize:
        toRight = 0
    fromLeft = ranks.index(rank) - 1
    if fromLeft < 0:
        fromLeft = csize - 1

    for step in range(csize):
        getFrom = ranks[fromLeft]
        sendTo = ranks[toRight]

        #send_buf = np.zeros(1)
        #send_buf[0] = rank

        #getSize = np.zeros(1, dtype=np.int32)
        #sendSize = np.zeros(1, dtype=np.int32)
        #sendSize = 1 if toRight == 1 else 0
        #sendSize[0] = len(send_buf)

        req = comm.irecv(source=getFrom, tag=TAG_CNT_PACKETS)
        size_send = 1 if toRight == 1 else 0
        comm.send(size_send, dest=sendTo, tag=TAG_CNT_PACKETS)

        get_size = req.wait()

        send_buf = [np.ones((2, 2))]
        #send_buf = data_source.get_data_for_receiver(toRight, len(ranks_receivers), pad)

        recv_buf = bytearray(1 << 20)
        if get_size > 0:
            req = comm.irecv(buf=recv_buf, source=getFrom, tag=TAG_PAYLOAD)

        if len(send_buf):
            comm.send(send_buf, dest=sendTo, tag=TAG_PAYLOAD)

        if get_size:
            recvData = req.wait()
            dst += [recvData]

        toRight += 1
        if toRight == csize:
            toRight = 0
        fromLeft -= 1
        if fromLeft < 0:
            fromLeft = csize - 1

    print(f"worker {comm.Get_rank()}   done")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.Barrier()
    if rank == 0:
        print("############################# STAGE 2 ##############################")
    logging.basicConfig(level=logging.DEBUG)
    local_data = []
    if rank == 0:
        local_data = [1, 2, 3, 4, 5, 6, 7]
    #if rank % 8 == 0:
     #   local_data = [np.random.random((2, 2)) for i in range(10)]
    comm.Barrier()

    #received_payload = np.zeros(0)
    received_payload = []
    shuffle(local_data, received_payload, comm,  pad=True, count_me_in=(rank ==1))
    comm.Barrier()
    #print(f"rank {rank}   received  {len(received_payload)}")
    print(f"rank {rank}   received  {received_payload}")
    comm.Barrier()
    if rank == 0:
        print(f"done!")


if __name__ == "__main__":
    main()

