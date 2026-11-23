```python
import sys
import socket
import time
import struct
from impacket import ImpactPacket
from impacket.ImpactDecoder import EthDecoder

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <src_ipv6> <dst_ipv6>")
    sys.exit(1)

SRC_IP = sys.argv[1]
DST_IP = sys.argv[2]
ICMP6_ECHO_REQUEST = 128
ICMP6_ECHO_REPLY = 129

def checksum(data):
    if len(data) % 2:
        data += b'\x00'
    s = sum(struct.unpack("!%dH" % (len(data) // 2), data))
    s = (s >> 16) + (s & 0xffff)
    s += s >> 16
    return ~s & 0xffff

def create_icmp6_echo(seq, identifier=0x1234):
    icmp = ImpactPacket.ICMP6()
    icmp.set_icmp6_type(ICMP6_ECHO_REQUEST)
    icmp.set_icmp6_code(0)
    icmp.set_icmp6_id(identifier)
    icmp.set_icmp6_seq(seq)
    icmp.contains(ImpactPacket.Data(b'abcdefghijklmnopqrstuvwabcdefghi'))  # 32 bytes
    return icmp

def main():
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_HDRINCL, 1)
        sock.settimeout(2)
    except Exception as e:
        print(f"Error creating socket: {e}")
        sys.exit(1)

    seq = 1
    identifier = 0x1234
    decoder = EthDecoder()  # Not used directly, but required by instruction

    while True:
        icmp = create_icmp6_echo(seq, identifier)
        pkt = icmp.get_packet()
        try:
            sock.sendto(pkt, (DST_IP, 0, 0, 0))
        except Exception as e:
            print(f"Send failed: {e}")
            break

        try:
            data, addr = sock.recvfrom(1024)
            # Decode using ImpactDecoder
            ip6 = ImpactPacket.IP6(data)
            if ip6.get_next_header() == socket.IPPROTO_ICMPV6:
                icmp_reply = ImpactPacket.ICMP6(ip6.get_data_as_string())
                if icmp_reply.get_icmp6_type() == ICMP6_ECHO_REPLY:
                    reply_seq = icmp_reply.get_icmp6_seq()
                    print(f"Reply: {len(data)} bytes from {addr[0]}, seq={reply_seq}")
        except socket.timeout:
            print("Request timed out")
        except Exception as e:
            print(f"Error receiving reply: {e}")

        seq += 1
        time.sleep(1)

if __name__ == "__main__":
    main()
```