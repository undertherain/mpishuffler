import mpi4py
from mpi4py import MPI
import numpy as np
import threading
import logging
import sys

sys_excepthook = sys.excepthook
def mpi_excepthook(v, t, tb):
    sys_excepthook(v, t, tb)
    mpi4py.MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


TAG_CNT_PACKETS = 11
TAG_PAYLOAD = 12

mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "multiple"


class MPILogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.comm = MPI.COMM_WORLD

    def debug(self, msg):
        #if self.comm.Get_rank() == 0:
        #self.logger.debug(msg)
        print(msg)


logger = MPILogger()


def get_cnt_samples_per_worker(size_data, cnt_shares):
    return (size_data + cnt_shares - 1) // cnt_shares


def get_ids_per_receiver(id_worker, cnt_samples_per_worker, cnt_workers, size_data, pad):
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


class ThreadReceiv(threading.Thread):
    def __init__(self, comm, dest):
        threading.Thread.__init__(self)
        self.comm = comm
        self.dest = dest
        logger.debug(f"receiving thread initilized on rank {self.comm.Get_rank()}")

    def run(self):
        cnt_sizes = 0
        cnt_packets = 0
        try:
            while True:
                status = MPI.Status()
                print(f"receiver {self.comm.Get_rank()} waiting, got {cnt_sizes} size updates from total of {self.comm.size}")
                buf = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                rank_sender = status.Get_source()
                print(f"receiver {self.comm.Get_rank()} got tag {tag} from {rank_sender}")
                if tag == TAG_CNT_PACKETS:
                    cnt_packets += buf
                    cnt_sizes += 1
                    print(f"receiver {self.comm.Get_rank()} got notification of {buf} more packets from {rank_sender}")
                else:
                    self.dest += buf
                    cnt_packets -= 1
                    print(f"receiver {self.comm.Get_rank()} got package of size {len(buf)} from {rank_sender}")
                if (cnt_sizes >= self.comm.size) and (cnt_packets == 0):
                    break
        except Exception:
            exc = sys.exc_info()
            exc_type, exc_obj, exc_trace = exc
            print("exception in receiver thread:")
            print(exc_type, exc_obj)
            print(exc_trace)
        print(f"receiver {self.comm.Get_rank()}  done")


class ThreadSend(threading.Thread):
    def __init__(self, comm, cnt_samples_per_worker, data_source, pad, receivers):
        threading.Thread.__init__(self)
        self.comm = comm
        self.cnt_samples_per_worker = cnt_samples_per_worker
        self.data_source = data_source
        self.pad = pad
        self.receivers = receivers
        logger.debug(f"sender thread  {self.comm.Get_rank()} initilized")

    def run(self):
        try:
            cnt_receivers = len(self.receivers)
            # print("receivers: ", self.receivers)
            for id_receiver in range(len(self.receivers)):
                ids = get_ids_per_receiver(id_receiver, self.cnt_samples_per_worker, cnt_receivers, self.data_source.size_global, self.pad)
                # logger.debug(f"sender {self.comm.Get_rank()} , ids  for rcver {id_receiver} : {ids}")
                lo, hi = get_local_subindices(ids, self.data_source.lo, self.data_source.hi)
                send_buf = []
                #logger.debug(f"sender {self.comm.Get_rank()}  subindices for rcver {id_receiver} as {lo}-{hi}, ids = {ids}")
                #print(self.data_source.data)
                if lo < hi:
                    send_buf = [self.data_source.data[i - self.data_source.lo] for i in ids[lo:hi]]
                # print(f"send buf on {self.comm.Get_rank()}: {send_buf}")
                size_packet = 100
                cnt_packets = get_cnt_samples_per_worker(len(send_buf), size_packet)
                logger.debug(f"sender {self.comm.Get_rank()} sending {cnt_packets} packets to {self.receivers[id_receiver]}")
                self.comm.ssend(cnt_packets, dest=self.receivers[id_receiver], tag=TAG_CNT_PACKETS)
                for id_packet in range(cnt_packets):
                    packet = send_buf[id_packet * size_packet: (id_packet + 1) * size_packet]
                    logger.debug(f"sender {self.comm.Get_rank()} sending packet of size {len(packet)} to {self.receivers[id_receiver]}")
                    self.comm.send(packet, dest=self.receivers[id_receiver], tag=TAG_PAYLOAD)
                logger.debug(f"sender {self.comm.Get_rank()} done sending {cnt_packets} packets to {self.receivers[id_receiver]}")
            print(f"sender {self.comm.Get_rank()}   done")
        except Exception:
            exc = sys.exc_info()
            exc_type, exc_obj, exc_trace = exc
            print("exception in sender thread:")
            print(exc_type, exc_obj)
            print(exc_trace)


def shuffle(src, dst, comm, pad=False, count_me_in=True):
    ranks_receivers = comm.allgather(comm.Get_rank() if count_me_in else -1)
    ranks_receivers = [i for i in ranks_receivers if i >= 0]
    data_source = DataSource(src, comm)
    cnt_samples_per_receiver = get_cnt_samples_per_worker(data_source.size_global, len(ranks_receivers))
    # logger.debug(f"samples per worker = {cnt_samples_per_receiver}")
    if count_me_in:
        receiver = ThreadReceiv(comm, dst)
        receiver.start()
    sender = ThreadSend(comm, cnt_samples_per_receiver, data_source, pad, ranks_receivers)
    sender.start()
    if count_me_in:
        receiver.join()
    sender.join()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    received_payload = []
    local_data = []
    #if rank == 0:
        #local_data = ["apple", "banana", "dekopon"]
    if rank == 1:
        local_data = ["a"] * 100000
    received_payload = []
    #shuffle(local_data, received_payload, comm, pad=False, count_me_in=(rank % 8 == 0))

    comm.Barrier()
    if rank == 0:
        print("############################# STAGE 2 ##############################")
    logging.basicConfig(level=logging.DEBUG)
    local_data = []
    if rank % 8 == 0:
        local_data = [np.random.random((100, 100)) for i in range(1000)]
    comm.Barrier()

    received_payload = []
    shuffle(local_data, received_payload, comm, pad=True, count_me_in=True)
    # shuffle(local_data, received_payload, comm, pad=True, count_me_in=True)
    comm.Barrier()
    print(f"rank {rank}   received  {len(received_payload)}")
    comm.Barrier()
    if rank  ==  0:
        print(f"done!")



if __name__ == "__main__":
    main()
