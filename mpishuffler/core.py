from mpi4py import MPI
import numpy as np
import threading
import logging



class MPILogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.comm = MPI.COMM_WORLD
    
    def debug(self, msg):
        if self.comm.Get_rank() == 0:
            self.logger.debug(msg)


logger = MPILogger()


def get_cnt_sample_per_worker(size_data, cnt_workers):
    return (size_data + cnt_workers - 1) // cnt_workers


def get_ids_per_receiver(id_worker, cnt_samples_per_worker, cnt_workers, size_data, pad):
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


class ThreadReceiv(threading.Thread):
    def __init__(self, comm, dest):
        threading.Thread.__init__(self)
        self.comm = comm
        self.dest = dest
        logger.debug("receiving thread initilized")

    def run(self):
        for i in range(self.comm.size):
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            self.dest += (data)


class ThreadSend(threading.Thread):
    def __init__(self, comm, cnt_samples_per_worker, data_source, pad, receivers):
        threading.Thread.__init__(self)
        self.comm = comm
        self.cnt_samples_per_worker = cnt_samples_per_worker
        self.data_source = data_source
        self.pad = pad
        self.receivers = receivers
        logger.debug("sending thread initilized")

    def run(self):
        cnt_receivers = len(self.receivers)
        for id_receiver in range(cnt_receivers):
            ids = get_ids_per_receiver(id_receiver, self.cnt_samples_per_worker, cnt_receivers, self.data_source.size_global, self.pad)
            # print("computed ids receiver")
            lo, hi = get_local_subindices(ids, self.data_source.lo, self.data_source.hi)
            # print(f"sender {self.comm.Get_rank()} computed local subindices as {lo}-{hi}")
            # if id_worker == 1 and self.comm.Get_rank()==0:
                # print(f"worker {id_worker} needs ids {ids}")
                # print(f"sender {self.comm.Get_rank()} has ids {self.data_source.lo} to {self.data_source.hi}")
                # print(f"worker {id_worker} sub lo, hi =  {lo}, {hi}")
                # print(f"sender {self.comm.Get_rank()} sending to {id_worker}, lo = {lo}, hi={hi}")
            send_buf = []
            if lo < hi:
                send_buf = [self.data_source.data[i - self.data_source.lo] for i in ids[lo:hi]]
                #for i in ids[lo:hi]:
                    #send_buf.append(self.data_source.data[i - self.data_source.lo])
            print(f"{self.comm.Get_rank()} about to send {len(send_buf)} to {self.receivers[id_receiver]}")
            # TODO: frist send a number of packages, then send packagew one by one
            self.comm.send(send_buf, dest=self.receivers[id_receiver])


def get_ranks_receivers():
    return 7


def shuffle(src, dst, comm, pad=False, count_me_in=True):
    ranks_receivers = comm.allgather(comm.Get_rank() if count_me_in else -1)
    ranks_receivers = [i for i in ranks_receivers if i >= 0]
    data_source = DataSource(src, comm)
    cnt_samples_per_receiver = get_cnt_sample_per_worker(data_source.size_global, len(ranks_receivers))
    logger.debug(f"samples per worker = {cnt_samples_per_receiver}")
    if count_me_in:
        receiver = ThreadReceiv(comm, dst)
        receiver.start()
    sender = ThreadSend(comm, cnt_samples_per_receiver, data_source, pad, ranks_receivers)
    sender.start()
    if count_me_in:
        receiver.join()
    sender.join()


def main():
    logging.basicConfig(level=logging.DEBUG)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    local_data = np.array([], dtype=np.int32)
#    if rank == 0:
#        local_data = ["apple", "banana", "dekopon"]
    if rank == 0:
        local_data = [np.random.random((100,100)) for i in range(55000)]
    #if rank == 1:
    #    local_data = np.arange(3)

    comm.Barrier()

    received_payload = []
    shuffle(local_data, received_payload, comm, pad=False, count_me_in=(rank<2))
    # shuffle(local_data, received_payload, comm, pad=False, count_me_in=True)
    comm.Barrier()
    print(f"rank {rank}   reveived  {len(received_payload)}")


if __name__ == "__main__":
    main()
