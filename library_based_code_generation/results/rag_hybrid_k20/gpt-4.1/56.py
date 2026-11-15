```python
from seed_emulator import (
    createEmulation,
    Emulator,
    createAutonomousSystem,
    getAutonomousSystem,
    getRouters,
    getNetworks,
    createInternetExchange,
    joinInternetExchange,
    setInternetExchange,
    setAutonomousSystem,
    __setUpIbgpMesh,
    build_emulator,
)

# Create the emulator
emulator = Emulator()

# Create AS150 (Transit AS) with 4 routers and 3 networks
as150 = createAutonomousSystem(emulator, asn=150, num_routers=4, num_networks=3)
setAutonomousSystem(emulator, as150)

# Create AS151 with 1 router, 1 network, and a web host
as151 = createAutonomousSystem(emulator, asn=151, num_routers=1, num_networks=1)
setAutonomousSystem(emulator, as151)

# Create AS152 with 1 router, 1 network, and a web host
as152 = createAutonomousSystem(emulator, asn=152, num_routers=1, num_networks=1)
setAutonomousSystem(emulator, as152)

# Create Internet Exchange
ix = createInternetExchange(emulator, "IX1")
setInternetExchange(emulator, ix)

# AS151 and AS152 join the Internet Exchange
joinInternetExchange(emulator, as151, ix)
joinInternetExchange(emulator, as152, ix)

# Set up BGP peering between AS150 and AS151
routers_as150 = getRouters(emulator, as150)
routers_as151 = getRouters(emulator, as151)
emulator.add_bgp_peering(routers_as150[0], routers_as151[0])

# Set up BGP peering between AS150 and AS152
routers_as152 = getRouters(emulator, as152)
emulator.add_bgp_peering(routers_as150[1], routers_as152[0])

# Add all layers to the emulator
build_emulator(emulator)

# Dump the emulator's state to a binary file
with open("emulator_state.bin", "wb") as f:
    f.write(emulator.dump_state())
```