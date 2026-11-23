```python
from seed_emulator import (
    createEmulation,
    Emulator,
    createAutonomousSystem,
    createInternetExchange,
)

# Create the emulator
emu = Emulator()

# Create AS150 (Transit AS)
as150 = createAutonomousSystem(asn=150, name="AS150")
# Add 4 routers to AS150
routers_as150 = [as150.add_router(f"r{i+1}") for i in range(4)]
# Add 3 networks to AS150
networks_as150 = [as150.add_network(f"net{i+1}") for i in range(3)]
# Connect routers to networks in AS150
as150.connect_router_to_network(routers_as150[0], networks_as150[0])
as150.connect_router_to_network(routers_as150[1], networks_as150[1])
as150.connect_router_to_network(routers_as150[2], networks_as150[2])
# Connect routers internally (full mesh)
for i in range(len(routers_as150)):
    for j in range(i+1, len(routers_as150)):
        as150.connect_routers(routers_as150[i], routers_as150[j])

# Create AS151
as151 = createAutonomousSystem(asn=151, name="AS151")
router_as151 = as151.add_router("r1")
network_as151 = as151.add_network("net1")
webhost_as151 = as151.add_host("web1")
as151.connect_router_to_network(router_as151, network_as151)
as151.connect_host_to_network(webhost_as151, network_as151)

# Create AS152
as152 = createAutonomousSystem(asn=152, name="AS152")
router_as152 = as152.add_router("r1")
network_as152 = as152.add_network("net1")
webhost_as152 = as152.add_host("web1")
as152.connect_router_to_network(router_as152, network_as152)
as152.connect_host_to_network(webhost_as152, network_as152)

# Create Internet Exchange and connect AS151 and AS152
ix = createInternetExchange("IX1")
ix.add_member(router_as151)
ix.add_member(router_as152)

# Add ASes and IX to emulator
emu.add_layer(as150)
emu.add_layer(as151)
emu.add_layer(as152)
emu.add_layer(ix)

# Set up BGP peering between AS150 and AS151, and AS150 and AS152
emu.add_ebgp_peering(routers_as150[0], router_as151)
emu.add_ebgp_peering(routers_as150[1], router_as152)

# Dump emulator state to binary file
emu.dump_state("emulator_state.bin")
```