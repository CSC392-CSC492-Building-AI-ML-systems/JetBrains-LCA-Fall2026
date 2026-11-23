```python
import sys
import socket
import time
from impacket import ImpactPacket
from impacket.ImpactDecoder import EthDecoder

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <src_ipv6> <dst_ipv6>")
    sys.exit(1)

src_ip = sys.argv[1]
dst_ip = sys.argv[2]

ICMP6_ECHO_REQUEST_TYPE = 128
ICMP6_ECHO_REPLY_TYPE = 129

def create_icmp6_echo(seq):
    icmp6 = ImpactPacket.ICMP6()
    icmp6.set_icmp_type(ICMP6_ECHO_REQUEST_TYPE)
    icmp6.set_icmp_code(0)
    icmp6.set_icmp_seq(seq)
    icmp6.contains(ImpactPacket.Data(b'abcdefghijklmnopqrstuvwabcdefghi'))  # 32 bytes
    icmp6.set_icmp_cksum(0)
    return icmp6

def create_ipv6_packet(src, dst, payload):
    ip6 = ImpactPacket.IP6()
    ip6.set_source_address(src)
    ip6.set_destination_address(dst)
    ip6.contains(payload)
    return ip6

def main():
    seq = 0
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6)
        s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_HDRINCL, 1)
        s.bind((src_ip, 0))
    except Exception as e:
        print(f"Socket error: {e}")
        sys.exit(1)

    decoder = EthDecoder('linux')

    while True:
        seq += 1
        icmp6 = create_icmp6_echo(seq)
        ip6 = create_ipv6_packet(src_ip, dst_ip, icmp6)
        pkt = ip6.get_packet()
        try:
            s.sendto(pkt, (dst_ip, 0, 0, 0))
        except Exception as e:
            print(f"Send error: {e}")
            continue

        try:
            s.settimeout(2)
            reply, addr = s.recvfrom(65535)
        except socket.timeout:
            print("Request timed out.")
            continue
        except Exception as e:
            print(f"Recv error: {e}")
            continue

        # Decode the reply
        try:
            ip6_reply = ImpactPacket.IP6(reply)
            icmp6_reply = ip6_reply.child()
            if icmp6_reply.get_icmp_type() == ICMP6_ECHO_REPLY_TYPE:
                size = len(reply)
                dst_addr = ip6_reply.get_source_address()
                seq_num = icmp6_reply.get_icmp_seq()
                print(f"Reply: {size} bytes from {dst_addr}, seq={seq_num}")
        except Exception as e:
            print(f"Decode error: {e}")

        time.sleep(1)

if __name__ == "__main__":
    main()
```