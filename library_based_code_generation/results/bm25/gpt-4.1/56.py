```python
from seed_emulator import (
    createEmulation,
    makeStubAsWithHosts,
    getNodeByAsnAndName,
    joinNetwork,
    joinInternetExchange,
    setInternetExchange,
    getNetworks,
    getRouters,
    InternetExchange,
    gen_emulation_files,
)

# Create the emulator
emu = createEmulation()

# Create AS150 (transit AS) with 4 routers and 3 networks
as150 = makeStubAsWithHosts(
    asn=150,
    num_routers=4,
    num_networks=3,
    num_hosts_per_network=0,
    transit=True
)
emu.add_layer(as150)

# Create AS151 with 1 router, 1 network, 1 web host
as151 = makeStubAsWithHosts(
    asn=151,
    num_routers=1,
    num_networks=1,
    num_hosts_per_network=1,
    host_type='web'
)
emu.add_layer(as151)

# Create AS152 with 1 router, 1 network, 1 web host
as152 = makeStubAsWithHosts(
    asn=152,
    num_routers=1,
    num_networks=1,
    num_hosts_per_network=1,
    host_type='web'
)
emu.add_layer(as152)

# Get routers and networks for each AS
as150_routers = getRouters(emu, 150)
as151_router = getRouters(emu, 151)[0]
as152_router = getRouters(emu, 152)[0]

as151_network = getNetworks(emu, 151)[0]
as152_network = getNetworks(emu, 152)[0]

# Create an Internet Exchange and have AS151 and AS152 join it
ix = InternetExchange("IX1")
setInternetExchange(emu, ix)
joinInternetExchange(emu, 151, ix)
joinInternetExchange(emu, 152, ix)

# Join AS151 and AS152 routers to their respective networks
joinNetwork(emu, as151_router, as151_network)
joinNetwork(emu, as152_router, as152_network)

# Set up BGP peering between AS150 and AS151, and between AS150 and AS152
# (Assume the first router of AS150 is used for peering)
as150_router1 = as150_routers[0]
as150_router2 = as150_routers[1]

emu.add_bgp_peering(as150_router1, as151_router)
emu.add_bgp_peering(as150_router2, as152_router)

# Add all layers to the emulator (already done above with add_layer)

# Dump the emulator's state to a binary file
gen_emulation_files(emu, "emulator_state.bin")
```