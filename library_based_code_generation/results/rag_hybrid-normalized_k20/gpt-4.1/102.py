```python
from seedemu import Emulator, Base, Routing, Ebgp, AutonomousSystem, createInternetExchange

def create_stub_as(base_layer, asn, exchange, num_hosts=2):
    # Create the AS
    asys = AutonomousSystem(asn)
    base_layer.addAutonomousSystem(asys)
    # Create a network for the AS
    net_name = f"net{asn}"
    net = base_layer.createNetwork(net_name)
    # Create hosts and add them to the network and AS
    for i in range(1, num_hosts + 1):
        host_name = f"as{asn}-host{i}"
        host = base_layer.createHost(host_name)
        net.addHost(host)
        asys.addHost(host)
    # Create a router, add to network and AS
    router_name = f"as{asn}-router"
    router = base_layer.createRouter(router_name)
    net.addRouter(router)
    asys.addRouter(router)
    # Join the router to the specified exchange
    base_layer.joinInternetExchange(router, exchange)
    return asys, router

# Create the emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
ebgp = Ebgp()

# Create internet exchanges
ix1 = createInternetExchange("IX1")
ix2 = createInternetExchange("IX2")
base.addInternetExchange(ix1)
base.addInternetExchange(ix2)

# Create core ASes
core_as1 = AutonomousSystem(65001)
core_as2 = AutonomousSystem(65002)
base.addAutonomousSystem(core_as1)
base.addAutonomousSystem(core_as2)

# Create networks for core ASes
core_net1 = base.createNetwork("core-net1")
core_net2 = base.createNetwork("core-net2")

# Create routers for core ASes
core_router1 = base.createRouter("core1-router")
core_router2 = base.createRouter("core2-router")
core_net1.addRouter(core_router1)
core_net2.addRouter(core_router2)
core_as1.addRouter(core_router1)
core_as2.addRouter(core_router2)

# Join core routers to exchanges
base.joinInternetExchange(core_router1, ix1)
base.joinInternetExchange(core_router2, ix2)

# Create stub ASes and connect them to exchanges
stub_as_list = []
stub_routers = []
for asn, exchange in [(65101, ix1), (65102, ix1), (65103, ix2), (65104, ix2)]:
    asys, router = create_stub_as(base, asn, exchange)
    stub_as_list.append(asys)
    stub_routers.append(router)

# Add private peering between stub ASes (e.g., between 65101 and 65103)
routing.addPrivatePeering(65101, 65103)
routing.addPrivatePeering(65102, 65104)

# Add layers to emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)

# Dump emulator state to binary file
emu.dump("emulation.bin")
```