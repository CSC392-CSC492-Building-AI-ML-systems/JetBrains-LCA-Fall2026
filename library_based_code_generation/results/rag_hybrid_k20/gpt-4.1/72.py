```python
from seed_emulator import (
    createEmulation,
    createAutonomousSystem,
    createInternetExchange,
    joinInternetExchange,
    addCrossConnectPeering,
    configureAsEvpnProvider,
    gen_emulation_files,
)

# Create the emulation
emu = createEmulation()

# Create AS150, AS151, AS152
as150 = createAutonomousSystem(150)
as151 = createAutonomousSystem(151)
as152 = createAutonomousSystem(152)

emu.add_autonomous_system(as150)
emu.add_autonomous_system(as151)
emu.add_autonomous_system(as152)

# Add 'web' host and 'router0' to each AS, and connect them to 'net0'
for asys in [as150, as151, as152]:
    net0 = asys.add_network('net0')
    web = asys.add_host('web')
    router0 = asys.add_router('router0')
    net0.add_member(web)
    net0.add_member(router0)
    # Install web service on 'web'
    web.install_service('web')

# Cross connection between AS150 and AS152 routers
addCrossConnectPeering(as150.get_router('router0'), as152.get_router('router0'))

# Create Internet Exchange 100
ix100 = createInternetExchange(100)
emu.add_internet_exchange(ix100)

# AS150 and AS151 join IX100 as peers
joinInternetExchange(as150, ix100)
joinInternetExchange(as151, ix100)

# AS150 is a provider for AS152
configureAsEvpnProvider(as150, as152)

# Render and compile using Docker with self-managed network
gen_emulation_files(
    emu,
    output_dir='./cross-connect',
    backend='docker',
    network_mode='self-managed'
)
```