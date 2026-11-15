```python
from seedemu import Emulator, Base, Routing, Ebgp, AutonomousSystem, createInternetExchange

def create_stub_as(base_layer, asn, exchange, num_hosts=2):
    # Create the AS
    asys = AutonomousSystem(asn)
    base_layer.addAutonomousSystem(asys)
    # Create a network for the AS
    net = base_layer.createNetwork(f"as{asn}-net")
    # Create hosts and add them to the network and AS
    for i in range(1, num_hosts + 1):
        host_name = f"as{asn}-host{i}"
        host = base_layer.createHost(host_name)
        base_layer.joinNetwork(host, net)
        asys.addHost(host)
    # Create a router, add to network and AS
    router = base_layer.createRouter(f"as{asn}-rtr")
    base_layer.joinNetwork(router, net)
    asys.addRouter(router)
    # Join the router to the specified exchange
    base_layer.joinInternetExchange(router, exchange)
    return asys, router

# Create emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
ebgp = Ebgp()

# Create internet exchanges
ix1 = createInternetExchange("IX1")
ix2 = createInternetExchange("IX2")
base.addInternetExchange(ix1)
base.addInternetExchange(ix2)

# Create multiple stub ASes
as1, rtr1 = create_stub_as(base, 65001, ix1)
as2, rtr2 = create_stub_as(base, 65002, ix1)
as3, rtr3 = create_stub_as(base, 65003, ix2)
as4, rtr4 = create_stub_as(base, 65004, ix2)
as5, rtr5 = create_stub_as(base, 65005, ix1)

# Add private peering relationships between ASes
ebgp.addPrivatePeering(65001, 65002)
ebgp.addPrivatePeering(65002, 65003)
ebgp.addPrivatePeering(65003, 65004)
ebgp.addPrivatePeering(65004, 65005)
ebgp.addPrivatePeering(65005, 65001)

# Add layers to emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)

# Dump emulator state to binary file
emu.dump("emulation.bin")
```