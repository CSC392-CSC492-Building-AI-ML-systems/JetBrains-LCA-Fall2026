```python
import sys
import socket
import time
import struct
from impacket import ImpactPacket
from impacket.ImpactDecoder import IP6Decoder

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src_ipv6> <dst_ipv6>")
        sys.exit(1)

    src_ip = sys.argv[1]
    dst_ip = sys.argv[2]

    # Create raw socket for ICMPv6
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6)
    except PermissionError:
        print("You need to run this script as root.")
        sys.exit(1)

    # Bind to source address
    sock.bind((src_ip, 0))

    decoder = IP6Decoder()

    identifier = 0xabcd
    sequence = 0

    while True:
        # Build ICMPv6 Echo Request
        icmp6 = ImpactPacket.ICMP6()
        icmp6.set_icmp_type(128)  # Echo Request
        icmp6.set_icmp_code(0)
        icmp6.contains(ImpactPacket.Data(b'PythonICMP6Ping'))

        # Set identifier and sequence number
        icmp6.set_echo_id(identifier)
        icmp6.set_echo_sequence(sequence)

        # Build IPv6 packet
        ip6 = ImpactPacket.IP6()
        ip6.set_source_address(src_ip)
        ip6.set_destination_address(dst_ip)
        ip6.contains(icmp6)

        # Send packet
        pkt = ip6.get_packet()
        sock.sendto(pkt, (dst_ip, 0, 0, 0))

        # Wait for reply
        try:
            sock.settimeout(2)
            reply, addr = sock.recvfrom(65535)
        except socket.timeout:
            print("Request timed out.")
            sequence += 1
            time.sleep(1)
            continue

        # Decode reply
        decoded = decoder.decode(reply)
        icmp_layer = decoded.child()
        if icmp_layer.get_icmp_type() == 129:  # Echo Reply
            seq = icmp_layer.get_echo_sequence()
            print(f"Reply from {addr[0]}: size={len(reply)} seq={seq}")
        else:
            print(f"Received ICMP type {icmp_layer.get_icmp_type()} from {addr[0]}")

        sequence += 1
        time.sleep(1)

if __name__ == "__main__":
    main()
```