```python
from seedemu import Emulator
from seedemu.layers import Base, Routing, Ebgp
from seedemu.services import DomainNameCachingService

# Create the emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
ebgp = Ebgp()

# Define ASes and IXes
ases = [100, 200, 300]
ixes = [1, 2]
hosts_per_as = 3

# Create IXes
for ix in ixes:
    base.createInternetExchange(ix)

# Create ASes, routers, hosts, and networks
as_routers = {}
as_hosts = {}

for asn in ases:
    net = base.createNetwork(asn)
    router = base.createRouter(asn, f"r{asn}")
    as_routers[asn] = router
    base.joinNetwork(asn, router, net)
    hosts = []
    for i in range(1, hosts_per_as + 1):
        host = base.createHost(asn, f"h{asn}-{i}")
        hosts.append(host)
        base.joinNetwork(asn, host, net)
    as_hosts[asn] = hosts

# Connect routers to IXes
base.joinInternetExchange(100, as_routers[100], 1)
base.joinInternetExchange(200, as_routers[200], 1)
base.joinInternetExchange(300, as_routers[300], 2)

# Install Domain Name Caching Service on specific hosts
dns_service = DomainNameCachingService()
dns_hosts = [
    (100, "h100-1"),
    (200, "h200-2"),
    (300, "h300-3"),
]
for asn, hostname in dns_hosts:
    dns_service.install(asn, hostname)
    dns_service.addBinding(asn, hostname)

# Add routing (OSPF within AS)
for asn in ases:
    routing.addRouter(asn, as_routers[asn])
    for host in as_hosts[asn]:
        routing.addHost(asn, host)

# Add eBGP peerings between ASes
ebgp.addPrivatePeering(100, 200)
ebgp.addPrivatePeering(200, 300)
ebgp.addPrivatePeering(100, 300)

# Add layers and services to the emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addService(dns_service)

# Dump the emulator state to a binary file
emu.dump("emulation.bin")
```