```python
from seedemu import Emulator
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Ospf, WebService
from seedemu.core import AutonomousSystem, InternetExchange, Host, RealWorldAutonomousSystem

# Create layers
base = Base()
routing = Routing()
ebgp = Ebgp()
ibgp = Ibgp()
ospf = Ospf()
web = WebService()

# Create Internet Exchanges with custom display names
ix1 = InternetExchange(100, displayName="IXP Alpha")
ix2 = InternetExchange(200, displayName="IXP Beta")
ix3 = InternetExchange(300, displayName="IXP Gamma")
base.addInternetExchange(ix1)
base.addInternetExchange(ix2)
base.addInternetExchange(ix3)

# Create Transit Autonomous Systems
as1 = AutonomousSystem(1)
as2 = AutonomousSystem(2)
as3 = AutonomousSystem(3)
base.addAutonomousSystem(as1)
base.addAutonomousSystem(as2)
base.addAutonomousSystem(as3)

# Create single-homed stub Autonomous Systems with various services
as10 = AutonomousSystem(10)
as11 = AutonomousSystem(11)
as12 = AutonomousSystem(12)
base.addAutonomousSystem(as10)
base.addAutonomousSystem(as11)
base.addAutonomousSystem(as12)

# Add hosts to stub ASes
host10 = Host('host10')
host11 = Host('host11')
host12 = Host('host12')
as10.addHost(host10)
as11.addHost(host11)
as12.addHost(host12)

# Add web services to stub ASes
web.addService('host10', 'web', 80)
web.addService('host11', 'web', 8080)
web.addService('host12', 'web', 8000)

# Add a host with a customized IP address to AS10
custom_host = Host('customHost')
custom_host.setIp4Address('10.10.10.42')
as10.addHost(custom_host)

# Create a real-world Autonomous System
real_as = RealWorldAutonomousSystem(64512, name="RealWorldAS")
base.addAutonomousSystem(real_as)

# Enable remote access to AS2's network
as2.enableRemoteAccess()

# Connect ASes to Internet Exchanges
ix1.addParticipant(as1)
ix1.addParticipant(as10)
ix2.addParticipant(as2)
ix2.addParticipant(as11)
ix3.addParticipant(as3)
ix3.addParticipant(as12)
ix1.addParticipant(real_as)

# Set up peering via a route server (IXP Alpha)
ebgp.addRouteServer(ix1.getId())
ebgp.addPeering(as1.getId(), ix1.getId())
ebgp.addPeering(as10.getId(), ix1.getId())
ebgp.addPeering(real_as.getId(), ix1.getId())

# Set up private peering with different peer relationships
ebgp.addPrivatePeering(as1.getId(), as2.getId(), relationship='transit')
ebgp.addPrivatePeering(as2.getId(), as3.getId(), relationship='peer')
ebgp.addPrivatePeering(as3.getId(), as10.getId(), relationship='customer')

# Add all layers to the emulator
emu = Emulator()
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addLayer(ibgp)
emu.addLayer(ospf)
emu.addLayer(web)

# Save the emulator to a component file
emu.save('emulation-component.bin')

# Render and compile the emulator
emu.render()
emu.compile()
```