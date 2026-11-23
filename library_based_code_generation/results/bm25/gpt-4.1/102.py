```python
from seedemu import Emulator, Base, Routing, Ebgp

def create_stub_as(base_layer, asn, exchange, num_hosts=2):
    # Create the autonomous system
    asys = base_layer.createAutonomousSystem(asn)
    # Create a network for the AS
    net = base_layer.createNetwork(f"net-as{asn}")
    # Create hosts and add them to the network
    hosts = base_layer.createHostsOnNetwork(net, [f"h{i}-as{asn}" for i in range(1, num_hosts+1)])
    # Create a router and add it to the network
    router = base_layer.createRouter(f"r-as{asn}")
    base_layer.joinNetwork(router, net)
    # Add the router to the AS
    asys.addNode(router)
    # Add hosts to the AS
    for host in hosts:
        asys.addNode(host)
    # Join the router to the specified exchange
    base_layer.joinInternetExchange(router, exchange)
    return asys

# Create the emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
ebgp = Ebgp()

# Create internet exchanges
ix1 = base.createInternetExchange("ix1")
ix2 = base.createInternetExchange("ix2")

# Create multiple stub ASes
as1 = create_stub_as(base, 65001, ix1)
as2 = create_stub_as(base, 65002, ix1)
as3 = create_stub_as(base, 65003, ix2)
as4 = create_stub_as(base, 65004, ix2)
as5 = create_stub_as(base, 65005, ix1)

# Add private peering relationships between ASes
ebgp.addPrivatePeering(65001, 65002)
ebgp.addPrivatePeering(65003, 65004)
ebgp.addPrivatePeering(65002, 65005)

# Add layers to the emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)

# Dump the emulator state to a binary file
emu.dump("emulation.bin")
```