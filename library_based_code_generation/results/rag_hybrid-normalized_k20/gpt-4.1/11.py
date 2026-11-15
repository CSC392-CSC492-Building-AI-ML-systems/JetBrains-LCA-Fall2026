```python
from seed_emulator import (
    createEmulation,
    createAutonomousSystem,
    createNetwork,
    createHostsOnNetwork,
    Emulator,
)

# Create the emulator
emulator = createEmulation()

# Base Layer
base_layer = []

# Autonomous System 1: 1 router, 5 hosts, 1 network
as1 = createAutonomousSystem("AS1")
net1 = createNetwork("net1")
router1 = as1.createRouter("r1")
router1.joinNetwork(net1)
hosts_as1 = []
for i in range(1, 6):
    host = as1.createHost(f"h{i}")
    host.joinNetwork(net1)
    hosts_as1.append(host)
base_layer.append(as1)

# Autonomous System 2: 3 routers, each on a different network
as2 = createAutonomousSystem("AS2")
routers_as2 = []
for i in range(1, 4):
    net = createNetwork(f"net2_{i}")
    router = as2.createRouter(f"r{i}")
    router.joinNetwork(net)
    routers_as2.append(router)
base_layer.append(as2)

# Autonomous System 3: 2 routers, both on the same network
as3 = createAutonomousSystem("AS3")
net3 = createNetwork("net3")
router3_1 = as3.createRouter("r1")
router3_2 = as3.createRouter("r2")
router3_1.joinNetwork(net3)
router3_2.joinNetwork(net3)
base_layer.append(as3)

# Routing Layer (empty for this example, but created for completeness)
routing_layer = []

# eBGP Layer: Add private peering between different ASes
ebgp_layer = []

# Peering between AS1 router and AS2 routers
ebgp_layer.append(as1.addPrivatePeering(router1, as2, routers_as2[0]))
ebgp_layer.append(as1.addPrivatePeering(router1, as2, routers_as2[1]))
ebgp_layer.append(as1.addPrivatePeering(router1, as2, routers_as2[2]))

# Peering between AS2 routers and AS3 routers
ebgp_layer.append(as2.addPrivatePeering(routers_as2[0], as3, router3_1))
ebgp_layer.append(as2.addPrivatePeering(routers_as2[1], as3, router3_2))

# Add layers to emulator
emulator.addLayer("base", base_layer)
emulator.addLayer("routing", routing_layer)
emulator.addLayer("ebgp", ebgp_layer)

# Dump emulator state to a binary file
with open("emulator_state.bin", "wb") as f:
    emulator.dump(f)
```