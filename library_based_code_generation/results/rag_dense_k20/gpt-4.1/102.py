```python
from seedemu import Emulator, Base, Routing, Ebgp

def create_stub_as(base_layer, asn, exchange, num_hosts=2):
    # Create the autonomous system
    asys = base_layer.createAutonomousSystem(asn)
    # Create a network for the AS
    net = base_layer.createNetwork(f"as{asn}-net")
    # Create hosts and add them to the network and AS
    hosts = []
    for i in range(1, num_hosts + 1):
        host = base_layer.createHost(f"as{asn}-host{i}")
        base_layer.joinNetwork(host, net)
        base_layer.setAutonomousSystem(host, asn)
        hosts.append(host)
    # Create a router, add to network and AS
    router = base_layer.createRouter(f"as{asn}-rtr")
    base_layer.joinNetwork(router, net)
    base_layer.setAutonomousSystem(router, asn)
    # Join router to the specified exchange
    base_layer.joinInternetExchange(router, exchange)
    return {
        'asn': asn,
        'hosts': hosts,
        'router': router,
        'network': net,
        'exchange': exchange,
    }

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
ebgp.addPrivatePeering(as1['asn'], as2['asn'])
ebgp.addPrivatePeering(as2['asn'], as3['asn'])
ebgp.addPrivatePeering(as3['asn'], as4['asn'])
ebgp.addPrivatePeering(as4['asn'], as5['asn'])
ebgp.addPrivatePeering(as5['asn'], as1['asn'])

# Add layers to the emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)

# Dump the emulator state to a binary file
emu.dump("emulation.bin")
```