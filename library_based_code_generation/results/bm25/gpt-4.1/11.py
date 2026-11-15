```python
from seed_emulator import (
    createEmulation,
    createAutonomousSystem,
    createNetwork,
    createHostsOnNetwork,
    joinNetwork,
    setBaseSystem,
    setAutonomousSystem,
    getAutonomousSystem,
    createEmulatorBaseWith5StubASAndHosts,
    BaseSystem,
    shouldMerge,
)

from seed_emulator.layers import (
    BaseLayer,
    RoutingLayer,
    EBGP,
)

# Create the emulation
emulation = createEmulation()

# Base Layer
base_layer = BaseLayer()

# Autonomous System 1: 1 router, 5 hosts, all on one network
as1 = createAutonomousSystem("AS1")
net1 = createNetwork("net1")
router1 = as1.createRouter("r1")
joinNetwork(router1, net1)
hosts1 = createHostsOnNetwork(net1, 5)
for host in hosts1:
    as1.addHost(host)
base_layer.addAutonomousSystem(as1)

# Autonomous System 2: 3 routers, each on a different network
as2 = createAutonomousSystem("AS2")
net2 = createNetwork("net2")
net3 = createNetwork("net3")
net4 = createNetwork("net4")
router2_1 = as2.createRouter("r2_1")
router2_2 = as2.createRouter("r2_2")
router2_3 = as2.createRouter("r2_3")
joinNetwork(router2_1, net2)
joinNetwork(router2_2, net3)
joinNetwork(router2_3, net4)
base_layer.addAutonomousSystem(as2)

# Autonomous System 3: 2 routers, both on the same network
as3 = createAutonomousSystem("AS3")
net5 = createNetwork("net5")
router3_1 = as3.createRouter("r3_1")
router3_2 = as3.createRouter("r3_2")
joinNetwork(router3_1, net5)
joinNetwork(router3_2, net5)
base_layer.addAutonomousSystem(as3)

# Routing Layer
routing_layer = RoutingLayer()

# eBGP Layer with private peering between different ASes
ebgp_layer = EBGP()
ebgp_layer.addPrivatePeering(as1, as2)
ebgp_layer.addPrivatePeering(as2, as3)
ebgp_layer.addPrivatePeering(as1, as3)

# Add layers to the emulator
emulation.addLayer(base_layer)
emulation.addLayer(routing_layer)
emulation.addLayer(ebgp_layer)

# Dump emulator state to a binary file
with open("emulator_state.bin", "wb") as f:
    emulation.dump(f)
```