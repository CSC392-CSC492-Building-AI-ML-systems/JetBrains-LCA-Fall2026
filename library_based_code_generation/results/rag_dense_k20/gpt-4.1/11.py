```python
from seed_emulator import (
    createEmulation,
    build_emulator,
    createAutonomousSystem,
    createRealWorldRouter,
    Emulator,
    Ebgp,
)

# Create the emulator
emu = Emulator()

# Base Layer
base_layer = createEmulation("base")

# Autonomous System 1: 1 router, 5 hosts, all on one network
as1 = createAutonomousSystem("AS1")
router1 = createRealWorldRouter("as1-router1")
as1.add_router(router1)
network1 = as1.create_network("10.0.1.0/24")
router1.join_network(network1)
hosts_as1 = []
for i in range(1, 6):
    host = as1.create_host(f"as1-host{i}")
    host.join_network(network1)
    hosts_as1.append(host)
base_layer.add_autonomous_system(as1)

# Autonomous System 2: 3 routers, each on a different network
as2 = createAutonomousSystem("AS2")
routers_as2 = []
for i in range(1, 4):
    router = createRealWorldRouter(f"as2-router{i}")
    net = as2.create_network(f"10.0.2{i}.0/24")
    router.join_network(net)
    as2.add_router(router)
    routers_as2.append(router)
base_layer.add_autonomous_system(as2)

# Autonomous System 3: 2 routers, both on the same network
as3 = createAutonomousSystem("AS3")
network3 = as3.create_network("10.0.3.0/24")
router3_1 = createRealWorldRouter("as3-router1")
router3_2 = createRealWorldRouter("as3-router2")
router3_1.join_network(network3)
router3_2.join_network(network3)
as3.add_router(router3_1)
as3.add_router(router3_2)
base_layer.add_autonomous_system(as3)

# Routing Layer (empty, but included for completeness)
routing_layer = createEmulation("routing")

# eBGP Layer
ebgp_layer = createEmulation("eBGP")
ebgp = Ebgp()
# Add private peering between different ASes
ebgp.add_private_peering(as1, as2)
ebgp.add_private_peering(as2, as3)
ebgp.add_private_peering(as1, as3)
ebgp_layer.add_ebgp(ebgp)

# Add layers to emulator
emu.add_layer(base_layer)
emu.add_layer(routing_layer)
emu.add_layer(ebgp_layer)

# Build and dump emulator state to binary file
build_emulator(emu)
emu.dump_state("emulator_state.bin")
```