```python
import sys
import socket
import time
import struct
from impacket import ImpactPacket
from impacket.ImpactDecoder import EthDecoder, LinuxIp6Decoder

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

    decoder = LinuxIp6Decoder()

    seq = 0
    identifier = 0x1234

    while True:
        # Build ICMPv6 Echo Request
        icmp6 = ImpactPacket.ICMP6()
        icmp6.set_icmp6_type(128)  # Echo Request
        icmp6.set_icmp6_code(0)
        icmp6.set_icmp6_id(identifier)
        icmp6.set_icmp6_seq(seq)
        icmp6.contains(ImpactPacket.Data(b'abcdefghijklmnopqrstuvwabcdefghi'))  # 32 bytes

        # Build IPv6 packet
        ip6 = ImpactPacket.IP6()
        ip6.set_source_address(src_ip)
        ip6.set_destination_address(dst_ip)
        ip6.set_next_header(58)  # ICMPv6
        ip6.contains(icmp6)

        # Send packet
        sock.sendto(icmp6.get_packet(), (dst_ip, 0, 0, 0))

        # Wait for reply
        try:
            sock.settimeout(2)
            reply, addr = sock.recvfrom(65535)
        except socket.timeout:
            print("Request timed out.")
            seq += 1
            time.sleep(1)
            continue

        # Decode reply
        pkt = decoder.decode(reply)
        icmp_layer = None
        # Find ICMPv6 layer
        while pkt:
            if isinstance(pkt, ImpactPacket.ICMP6):
                icmp_layer = pkt
                break
            pkt = pkt.child()

        if icmp_layer and icmp_layer.get_icmp6_type() == 129:  # Echo Reply
            data = icmp_layer.child().get_bytes()
            print(f"Reply from {dst_ip}: size={len(data)} seq={icmp_layer.get_icmp6_seq()}")
        else:
            print("Received non-echo-reply packet.")

        seq = (seq + 1) & 0xffff
        time.sleep(1)

if __name__ == "__main__":
    main()
```