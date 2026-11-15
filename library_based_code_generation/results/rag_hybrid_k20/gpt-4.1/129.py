```python
import sys
import socket
import time
import struct
from impacket import ImpactPacket
from impacket.ImpactDecoder import IPDecoderForICMP

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src_ipv6> <dst_ipv6>")
        sys.exit(1)

    src_ip = sys.argv[1]
    dst_ip = sys.argv[2]

    # Create raw socket for ICMPv6
    s = socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6)
    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_HDRINCL, 1)

    decoder = IPDecoderForICMP()

    seq = 0
    identifier = 0x1234

    while True:
        # Build ICMPv6 Echo Request
        icmp6 = ImpactPacket.ICMP6()
        icmp6.set_icmp_type(ImpactPacket.ICMP6.ICMP6_ECHO_REQUEST)
        icmp6.set_icmp_code(0)
        icmp6.set_echo_identifier(identifier)
        icmp6.set_echo_sequence(seq)
        icmp6.contains(ImpactPacket.Data(b'ping6'))

        # Build IPv6 packet
        ip6 = ImpactPacket.IP6()
        ip6.set_source_address(src_ip)
        ip6.set_destination_address(dst_ip)
        ip6.contains(icmp6)

        # Send packet
        s.sendto(ip6.get_packet(), (dst_ip, 0, 0, 0))

        # Wait for reply
        while True:
            reply, addr = s.recvfrom(65535)
            pkt = decoder.decode(reply)
            icmp_layer = pkt.child()
            if icmp_layer.get_icmp_type() == ImpactPacket.ICMP6.ICMP6_ECHO_REPLY:
                if icmp_layer.get_echo_identifier() == identifier and icmp_layer.get_echo_sequence() == seq:
                    print(f"Reply from {dst_ip}: size={len(reply)} seq={icmp_layer.get_echo_sequence()}")
                    break
        seq = (seq + 1) & 0xffff
        time.sleep(1)

if __name__ == "__main__":
    main()
```