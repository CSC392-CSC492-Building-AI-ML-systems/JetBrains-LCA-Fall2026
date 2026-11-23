```python
from seedemu import *

# Create the emulator
emu = Emulator()

# Base Layer
base = Base()

# Create 3 Autonomous Systems (AS1, AS2, AS3)
as1 = AutonomousSystem(1)
as2 = AutonomousSystem(2)
as3 = AutonomousSystem(3)

# Create Internet Exchanges (IX1, IX2)
ix1 = InternetExchange(1)
ix2 = InternetExchange(2)

# Add ASes and IXes to base
base.addAutonomousSystem(as1)
base.addAutonomousSystem(as2)
base.addAutonomousSystem(as3)
base.addInternetExchange(ix1)
base.addInternetExchange(ix2)

# Add hosts and routers to each AS, connect to IX
for asys, ix in zip([as1, as2, as3], [ix1, ix2, ix1]):
    # Create a network in the AS
    net = asys.createNetwork('net')
    # Add 3 hosts
    h1 = asys.createHost('host1')
    h2 = asys.createHost('host2')
    h3 = asys.createHost('host3')
    # Add a router
    r = asys.createRouter('r')
    # Connect hosts and router to the network
    asys.joinNetwork('host1', 'net')
    asys.joinNetwork('host2', 'net')
    asys.joinNetwork('host3', 'net')
    asys.joinNetwork('r', 'net')
    # Connect router to IX
    asys.joinInternetExchange('r', ix.getId())

# Routing Layer
routing = Routing()
routing.addAutonomousSystem(1)
routing.addAutonomousSystem(2)
routing.addAutonomousSystem(3)

# eBGP Layer
ebgp = Ebgp()
# Add private peerings between AS1-AS2, AS2-AS3, AS1-AS3
ebgp.addPrivatePeering(1, 'r', 2, 'r')
ebgp.addPrivatePeering(2, 'r', 3, 'r')
ebgp.addPrivatePeering(1, 'r', 3, 'r')

# Domain Name Caching Service Layer
dnscache = DomainNameCachingService()
# Install DNS cache on host1 in AS1 and host2 in AS2
dnscache.install(1, 'host1')
dnscache.install(2, 'host2')
# Add bindings for the installations
dnscache.addBinding(1, 'host1')
dnscache.addBinding(2, 'host2')

# Add layers and services to the emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addService(dnscache)

# Dump the emulator state to a binary file
emu.dump('emulation.bin')
```