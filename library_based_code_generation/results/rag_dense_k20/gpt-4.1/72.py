```python
from seed_emulator import (
    createEmulation,
    createAutonomousSystem,
    createInternetExchange,
    addCrossConnectPeering,
    gen_emulation_files,
    Emulator,
)

# Create the emulation
emu = createEmulation()

# Create three ASes: 150, 151, 152
ases = {}
for asn in [150, 151, 152]:
    asys = createAutonomousSystem(asn)
    # Add router0
    router = asys.add_router('router0')
    # Add net0
    net = asys.add_network('net0')
    # Add web host and join net0
    web = asys.add_host('web')
    net.add_member(web)
    net.add_member(router)
    # Install web service on web host
    web.install_service('web')
    # Add AS to emulation
    emu.add_autonomous_system(asys)
    ases[asn] = asys

# Cross connection between AS150 and AS152 routers
addCrossConnectPeering(
    ases[150].get_router('router0'),
    ases[152].get_router('router0')
)

# Create Internet Exchange 100
ix = createInternetExchange(100)
emu.add_internet_exchange(ix)

# AS150 and AS151 are peers on IX100
ix.add_peer(ases[150])
ix.add_peer(ases[151])

# AS150 is provider for AS152
ases[150].add_customer(ases[152])

# Render and compile using Docker with self-managed network
gen_emulation_files(
    emu,
    output_dir='./cross-connect',
    backend='docker',
    docker_network_mode='self-managed'
)
```