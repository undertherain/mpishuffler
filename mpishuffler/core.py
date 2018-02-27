from mpi4py import MPI
import numpy as np
import threading


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

    def run(self):
        for i in range(self.comm.size):
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            self.dest += (data)


class ThreadSend(threading.Thread):
    def __init__(self, comm, cnt_samples_per_worker, data_source, pad):
        threading.Thread.__init__(self)
        self.comm = comm
        self.cnt_samples_per_worker = cnt_samples_per_worker
        self.data_source = data_source
        self.pad = pad

    def run(self):
        cnt_workers = self.comm.Get_size()
        for id_worker in range(cnt_workers):
            ids = get_ids_per_receiver(id_worker, self.cnt_samples_per_worker, cnt_workers, self.data_source.size_global, self.pad)

            lo, hi = get_local_subindices(ids, self.data_source.lo, self.data_source.hi)
            # print(f"sender {comm.Get_rank()}")
            # if id_worker == 1 and self.comm.Get_rank()==0:
                # print(f"worker {id_worker} needs ids {ids}")
                # print(f"sender {self.comm.Get_rank()} has ids {self.data_source.lo} to {self.data_source.hi}")
                # print(f"worker {id_worker} sub lo, hi =  {lo}, {hi}")
                # print(f"sender {self.comm.Get_rank()} sending to {id_worker}, lo = {lo}, hi={hi}")
            send_buf = []
            for i in ids[lo:hi]:
                send_buf.append(self.data_source.data[i - self.data_source.lo])
            self.comm.send(send_buf, dest=id_worker)


def redistribute(src, dst, comm, pad=False):
    cnt_receivers = comm.Get_size()
    data_source = DataSource(src, comm)
    cnt_samples_per_receiver = get_cnt_sample_per_worker(data_source.size_global, cnt_receivers)
    # print(f"samples per worker = {cnt_samples_per_worker}")
    receiver = ThreadReceiv(comm, dst)
    sender = ThreadSend(comm, cnt_samples_per_receiver, data_source, pad)
    receiver.start()
    sender.start()
    receiver.join()
    sender.join()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    local_data = np.array([], dtype=np.int32)
    if rank == 0:
        local_data = ["apple", "banana"]
    #if rank == 1:
    #    local_data = np.arange(3)

    comm.Barrier()

    received_payload = []
    redistribute(local_data, received_payload, comm, True)
    comm.Barrier()
    print(f"rank {rank}   reveived  {received_payload}")


if __name__ == "__main__":
    main()
