import subprocess
import socket
import time

class ModulateLatencyBandwidth(object):
    def __init__(self, interface, worker_rank, world_size, max_bw):
        self.interface = interface
        self.worker_rank = worker_rank
        self.world_size = world_size
        self.max_bw = max_bw
        self.ipv4_addr = socket.gethostbyname(socket.gethostname())

        #setup htb and qdisc on given interface
        cmd = f"sudo tc qdisc add dev {self.interface} root handle 1: htb default 10"
        cmd = [x for x in cmd.split()]
        subprocess.Popen(cmd)
        time.sleep(2.)

        # root class to represent total available b/w on a given NIC
        cmd2 = f'sudo tc class add dev {self.interface} parent 1: classid 1:1 htb rate {self.max_bw}gbit'
        cmd2 = [x for x in cmd2.split()]
        subprocess.Popen(cmd2)
        time.sleep(2.)

    def delteBWFilterLimits(self):
        cmd = f'sudo tc filter del dev {self.interface} protocol ip parent 1: prio 1 u32 match ip dst <server_IP> flowid 1:{self.world_size * (self.worker_rank + 1)}'
        cmd = [x for x in cmd.split()]
        subprocess.Popen(cmd)
        time.sleep(2.)


    def adjustAlphaBeta(self, latency, bandwidth):

        self.delteBWFilterLimits()

        split_bandwidth = bandwidth / self.world_size
        cmd = f'sudo tc class add dev {self.interface} parent 1:1 classid 1:{self.world_size * (self.worker_rank + 1)} htb rate {split_bandwidth}gbit'
        cmd = [x for x in cmd.split()]
        subprocess.Popen(cmd)
        time.sleep(2.)

        cmd2 = f'sudo tc qdisc add dev {self.interface} parent 1:{self.world_size * (self.worker_rank + 1)} handle 10: netem delay {latency}ms'
        cmd2 = [x for x in cmd2.split()]
        subprocess.Popen(cmd2)
        time.sleep(2.)

        cmd3 = f'sudo tc filter add dev {self.interface} protocol ip parent 1: prio 1 u32 match ip dst {self.ipv4_addr}0 flowid 1:{self.world_size * (self.worker_rank + 1)}'
        cmd3 = [x for x in cmd3.split()]
        subprocess.Popen(cmd3)
        time.sleep(2.)