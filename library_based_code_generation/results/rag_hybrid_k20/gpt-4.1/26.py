```python
from seedemu import Emulator, Base, Routing, Ebgp, Ibgp, Ospf, WebService
from seedemu.layers.base import AutonomousSystem, Host, InternetExchange
from seedemu.layers.routing import PeerRelationship

# Create the emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
ebgp = Ebgp()
ibgp = Ibgp()
ospf = Ospf()
web = WebService()

# Create Internet Exchanges with custom display names
ix1 = base.createInternetExchange(1, displayName="IX Frankfurt")
ix2 = base.createInternetExchange(2, displayName="IX Amsterdam")
ix3 = base.createInternetExchange(3, displayName="IX London")

# Create Transit Autonomous Systems
transit_as1 = base.createAutonomousSystem(100, displayName="Transit AS 100")
transit_as2 = base.createAutonomousSystem(200, displayName="Transit AS 200")

# Create single-homed stub Autonomous Systems with various services
stub_as1 = base.createAutonomousSystem(300, displayName="Stub AS 300")
stub_as2 = base.createAutonomousSystem(400, displayName="Stub AS 400")
stub_as3 = base.createAutonomousSystem(500, displayName="Stub AS 500")

# Add hosts to stub ASes
host1 = Host(name="web1", ip="10.0.0.10")
host2 = Host(name="db1")
host3 = Host(name="client1")
stub_as1.addHost(host1)
stub_as1.addHost(host2)
stub_as2.addHost(host3)

# Add a host with a customized IP address to one of the Autonomous Systems
custom_host = Host(name="specialhost", ip="192.168.100.100")
stub_as3.addHost(custom_host)

# Add web service to host1
web.addService("web1", "apache")

# Create a real-world Autonomous System (e.g., Google AS15169)
real_as = base.createAutonomousSystem(15169, displayName="Google")

# Connect ASes to Internet Exchanges
ix1.addParticipant(transit_as1)
ix1.addParticipant(stub_as1)
ix2.addParticipant(transit_as2)
ix2.addParticipant(stub_as2)
ix3.addParticipant(stub_as3)
ix3.addParticipant(real_as)

# Enable remote access to one of the Autonomous System's network
base.enableRemoteAccess(300)

# Set up peering via a route server at IX1
routing.addRouteServer(ix1.getId())
routing.addPeering(transit_as1.getId(), ix1.getId(), viaRouteServer=True)
routing.addPeering(stub_as1.getId(), ix1.getId(), viaRouteServer=True)

# Set up private peering with different peer relationships
routing.addPrivatePeering(transit_as1.getId(), transit_as2.getId(), PeerRelationship.PEER)
routing.addPrivatePeering(transit_as1.getId(), stub_as1.getId(), PeerRelationship.CUSTOMER)
routing.addPrivatePeering(transit_as2.getId(), stub_as2.getId(), PeerRelationship.CUSTOMER)
routing.addPrivatePeering(transit_as2.getId(), real_as.getId(), PeerRelationship.PEER)

# Add all layers to the emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addLayer(ibgp)
emu.addLayer(ospf)
emu.addLayer(web)

# Save the emulator to a component file
emu.save("emulation.bin")

# Render and compile the emulator
emu.render()
emu.compile()
```