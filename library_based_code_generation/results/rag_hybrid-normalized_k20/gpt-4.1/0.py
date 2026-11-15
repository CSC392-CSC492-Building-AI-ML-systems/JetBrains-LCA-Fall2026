```python
from seedemu import *

# Create the emulator
emu = Emulator()

# Base Layer
base = Base()

# Create 3 Autonomous Systems (AS1, AS2, AS3)
as1 = base.createAutonomousSystem(1)
as2 = base.createAutonomousSystem(2)
as3 = base.createAutonomousSystem(3)

# Create 2 Internet Exchanges (IX1, IX2)
ix1 = base.createInternetExchange(1)
ix2 = base.createInternetExchange(2)

# Add hosts and routers to each AS, join internal network and IX
for asys, ix in [(as1, ix1), (as2, ix1), (as3, ix2)]:
    # Create internal network
    net = asys.createNetwork('net')
    # Create router
    router = asys.createRouter('r1')
    net.addMember(router)
    # Create 2 hosts
    h1 = asys.createHost('h1')
    h2 = asys.createHost('h2')
    net.addMember(h1)
    net.addMember(h2)
    # Router joins IX
    ix.addMember(router)

# Routing Layer
routing = Routing()
routing.addAS(1)
routing.addAS(2)
routing.addAS(3)

# eBGP Layer
ebgp = Ebgp()
# Add private peerings between ASes
ebgp.addPrivatePeering(1, 2)
ebgp.addPrivatePeering(2, 3)
ebgp.addPrivatePeering(1, 3)

# Domain Name Caching Service Layer
dnscache = DomainNameCachingService()
# Install DNS cache on h1 in AS1 and h1 in AS2
dnscache.install(1, 'h1')
dnscache.install(2, 'h1')
# Add bindings for these installations
dnscache.addBinding(1, 'h1')
dnscache.addBinding(2, 'h1')

# Add layers and services to emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addServiceMerger(dnscache)

# Dump emulator state to binary file
emu.dump('emulation.bin')
```