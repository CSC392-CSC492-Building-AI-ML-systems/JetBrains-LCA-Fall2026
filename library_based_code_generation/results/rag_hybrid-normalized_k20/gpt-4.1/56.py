```python
from seed_emulator import (
    Emulator,
    createEmulation,
    setAutonomousSystem,
    createInternetExchange,
    joinInternetExchange,
    build_emulator,
)

# Create the emulator
emulator = Emulator()

# Create AS150 (Transit AS)
as150 = setAutonomousSystem(
    emulator,
    asn=150,
    routers=4,
    networks=3,
    is_transit=True
)

# Create AS151 (Stub AS)
as151 = setAutonomousSystem(
    emulator,
    asn=151,
    routers=1,
    networks=1,
    hosts_per_network=1,  # web host
    is_transit=False
)

# Create AS152 (Stub AS)
as152 = setAutonomousSystem(
    emulator,
    asn=152,
    routers=1,
    networks=1,
    hosts_per_network=1,  # web host
    is_transit=False
)

# Create an Internet Exchange
ix = createInternetExchange(emulator, name="IX1")

# AS151 and AS152 join the Internet Exchange
joinInternetExchange(emulator, asn=151, ix=ix)
joinInternetExchange(emulator, asn=152, ix=ix)

# Set up BGP peering between AS150 and AS151
emulator.add_bgp_peering(150, 151)

# Set up BGP peering between AS150 and AS152
emulator.add_bgp_peering(150, 152)

# Add all layers to the emulator (networks, routers, hosts, IX)
build_emulator(emulator)

# Dump the emulator's state to a binary file
with open("emulator_state.bin", "wb") as f:
    f.write(emulator.dump_state())
```