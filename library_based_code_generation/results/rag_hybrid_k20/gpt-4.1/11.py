```python
from seed_emulator import (
    createEmulation,
    createAutonomousSystem,
    createNetwork,
    createHostsOnNetwork,
    getAutonomousSystem,
    Ebgp,
    build_emulator,
)

# Create the emulation object
emulator = createEmulation("my_emulation")

# Base Layer
base_layer = []

# Autonomous System 1: 1 router, 5 hosts, 1 network
as1 = createAutonomousSystem("AS1")
net1 = createNetwork("net1")
router1 = as1.createRouter("r1")
router1.joinNetwork(net1)
hosts1 = createHostsOnNetwork(net1, count=5, prefix="h")
base_layer.append(as1)

# Autonomous System 2: 3 routers, each on a different network
as2 = createAutonomousSystem("AS2")
net2 = createNetwork("net2")
net3 = createNetwork("net3")
net4 = createNetwork("net4")
router2_1 = as2.createRouter("r2_1")
router2_2 = as2.createRouter("r2_2")
router2_3 = as2.createRouter("r2_3")
router2_1.joinNetwork(net2)
router2_2.joinNetwork(net3)
router2_3.joinNetwork(net4)
base_layer.append(as2)

# Autonomous System 3: 2 routers, both on the same network
as3 = createAutonomousSystem("AS3")
net5 = createNetwork("net5")
router3_1 = as3.createRouter("r3_1")
router3_2 = as3.createRouter("r3_2")
router3_1.joinNetwork(net5)
router3_2.joinNetwork(net5)
base_layer.append(as3)

# Routing Layer (empty for this example, but could be extended)
routing_layer = []

# eBGP Layer: private peering between different ASes
ebgp_layer = []
ebgp1 = Ebgp()
ebgp1.addPrivatePeering(as1, as2)
ebgp1.addPrivatePeering(as2, as3)
ebgp1.addPrivatePeering(as1, as3)
ebgp_layer.append(ebgp1)

# Add layers to emulator
emulator.addLayer("base", base_layer)
emulator.addLayer("routing", routing_layer)
emulator.addLayer("ebgp", ebgp_layer)

# Dump emulator state to a binary file
with open("emulator_state.bin", "wb") as f:
    f.write(emulator.dumpState())
```