- **Relevant links:**
    - **Traffic control (tc)**:
        - https://man7.org/linux/man-pages/man8/tc.8.html
        - https://www.techrepublic.com/article/how-to-limit-bandwidth-on-linux-to-better-test-your-applications/
        
    - **Measuring bandwidth and network performance (iperf)**:
        - https://netbeez.net/blog/how-to-use-the-linux-traffic-control/
        - https://www.cyberithub.com/iperf-commands-how-to-use-iperf-in-linux/

- **See current traffic control state:**
    - ```tc qdisc show```
    - ```tc qdisc show dev *interface*```
   
- **Set b/w on given interface:**
    - ```sudo tc qdisc add dev *eth0* root tbf rate 100kbit latency 2ms burst 1540```
    
- **Add delay on an interface:**
    - ```sudo tc qdisc add dev *eth0* root netem delay 200ms```
    (where **qdisc**: modify the scheduler, **add**: add a new rule, **root**: modify outbound traffic scheduler (egress qdisc),
    **netem**: use network emulator to emulate a WAN property, **delay**: add a delay in the network)
    
- **Clear all existing qdisc configurations:**
    - ```sudo tc qdisc del dev *eth0* root```
    
- **TO CHECK EFFECTIVE BANDWIDTH, USE iperf:**
    - **_On TCP port_**:
        - _On server-side_: ```iperf -s -f K```
    - **_On UDP port_**:
        - _On server-side_: ```iperf -s -u```
    - _On client-side_: ```iperf -c <ip addr of corresponding interface>```
    
- **Example config: splitting 40Gbps among 4 workers with delay 1ms**
 - Set up the htb (hierarchical token bucket) qdisc (queueing discipline) on the interface to control traffic:
    - ```sudo tc qdisc add dev ens6 root handle 1: htb default 10```
 
 - Create a root class that represents the total available bandwidth of the NIC:
    - ```sudo tc class add dev ens6 parent 1: classid 1:1 htb rate 40gbit```
 
 - Create individual classes for each server and set the bandwidth limit to 10 Gbps for each class:
    - ```sudo tc class add dev ens6 parent 1:1 classid 1:10 htb rate 10gbit```
    - ```sudo tc qdisc add dev ens6 parent 1:10 handle 10: netem delay 1ms```
    - ```sudo tc class add dev ens6 parent 1:1 classid 1:20 htb rate 10gbit```
    - ```sudo tc qdisc add dev ens6 parent 1:20 handle 10: netem delay 1ms```
    - ```sudo tc class add dev ens6 parent 1:1 classid 1:30 htb rate 10gbit```
    - ```sudo tc qdisc add dev ens6 parent 1:30 handle 10: netem delay 1ms```
    - ```sudo tc class add dev ens6 parent 1:1 classid 1:40 htb rate 10gbit```
    - ```sudo tc qdisc add dev ens6 parent 1:40 handle 10: netem delay 1ms```