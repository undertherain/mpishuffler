from mpi4py import MPI
from mpishuffler import shuffle


def load_super_huge_data():
    return ["apple", "banana", "dekopon"]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    local_data = []
    if rank == 0:
        local_data = load_super_huge_data()
    received_payload = []
    shuffle(local_data, received_payload, comm)
    print(f"rank {rank} reveived  {received_payload}")


if __name__ == "__main__":
    main()
