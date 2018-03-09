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


def shuffle(src, dst, comm, csize, rank, pad=False, count_me_in=True):
    status = MPI.Status()
    ranks = comm.allgather(comm.Get_rank())
    data_source = DataSource(src, comm)

    toRight = ranks.index(rank) + 1
    if toRight == csize: toRight = 0
    fromLeft = ranks.index(rank) - 1
    if fromLeft < 0: fromLeft = csize - 1

    cnt_samples_per_worker = get_cnt_samples_per_worker(data_source.size_global, csize)

    for step in range(csize - 1):
        getFrom = ranks[fromLeft]
        sendTo = ranks[toRight]
        # prepare data
        #ids = get_ids_per_receiver(sendTo, cnt_samples_per_worker, csize,
        #                           data_source.size_global, pad)
        #lo, hi = get_local_subindices(ids, data_source.lo, data_source.hi)
        send_buf = np.zeros(1)
        send_buf[0] = rank
        #if lo < hi:
        #    send_buf = [data_source.data[i - data_source.lo] for i in ids[lo:hi]]

        getSize = np.zeros(1)
        sendSize = np.zeros(1)
        sendSize[0] = len(send_buf)

        req = comm.Irecv(buf=getSize, source=getFrom, tag=TAG_CNT_PACKETS)
        comm.Send(buf=sendSize, dest=sendTo, tag=TAG_CNT_PACKETS)

        req.Wait()

        #recvData = np.zeros((100,3,100,100))
#        send_buf = np.zeros((10,3,100,100))
        send_buf = np.zeros((100, 100))
        #send_buf = [np.zeros((3, 100, 100))] * 10
        if getSize[0]:
            req = comm.irecv(source=getFrom, tag=TAG_PAYLOAD)
            # req = comm.Irecv(buf=recvData, source=getFrom, tag=TAG_PAYLOAD)

        if len(send_buf):
            comm.send(send_buf, dest=sendTo, tag=TAG_PAYLOAD)
            #comm.Send(send_buf, dest=sendTo, tag=TAG_PAYLOAD)

        if getSize[0]:
            # req.Wait()
            recvData = req.Wait()
            dst += [recvData]

        toRight += 1
        if toRight == csize: toRight = 0
        fromLeft -= 1
        if fromLeft < 0: fromLeft = csize - 1

    print(f"getFrom {comm.Get_rank()}   done")


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

    #received_payload = np.zeros(0)
    received_payload = []
    shuffle(local_data, received_payload, comm, csize, rank, pad=True, count_me_in=(rank ==1))
    comm.Barrier()
    print(f"rank {rank}   received  {len(received_payload)}")
    comm.Barrier()
    if rank == 0:
        print(f"done!")


if __name__ == "__main__":
    main()

