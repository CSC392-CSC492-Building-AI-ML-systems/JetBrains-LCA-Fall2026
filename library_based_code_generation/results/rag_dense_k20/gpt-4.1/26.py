```python
from seedemu import (
    Emulator, Base, Routing, Ebgp, Ibgp, Ospf, WebService,
    createInternetExchange, createAutonomousSystem, createRealWorldRouter
)

# Create emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
ebgp = Ebgp()
ibgp = Ibgp()
ospf = Ospf()
web = WebService()

# Create Internet Exchanges with custom display names
ix1 = createInternetExchange(display_name="IX Frankfurt")
ix2 = createInternetExchange(display_name="IX Amsterdam")
ix3 = createInternetExchange(display_name="IX London")
base.addInternetExchange(ix1)
base.addInternetExchange(ix2)
base.addInternetExchange(ix3)

# Create Transit Autonomous Systems
transit_as1 = createAutonomousSystem(asn=100, name="Transit AS 100")
transit_as2 = createAutonomousSystem(asn=200, name="Transit AS 200")
base.addAutonomousSystem(transit_as1)
base.addAutonomousSystem(transit_as2)

# Create single-homed stub Autonomous Systems with various services
stub_as1 = createAutonomousSystem(asn=101, name="Stub AS 101")
stub_as2 = createAutonomousSystem(asn=102, name="Stub AS 102")
stub_as3 = createAutonomousSystem(asn=103, name="Stub AS 103")
base.addAutonomousSystem(stub_as1)
base.addAutonomousSystem(stub_as2)
base.addAutonomousSystem(stub_as3)

# Add services to stub ASes
web.addWebService(asn=101, hostName="web1", serviceName="nginx")
web.addWebService(asn=102, hostName="web2", serviceName="apache")
web.addWebService(asn=103, hostName="web3", serviceName="nginx")

# Add a host with a customized IP address to one of the Autonomous Systems
base.addHost(asn=101, name="customhost", ip="10.101.0.99")

# Create a real-world Autonomous System
real_as = createRealWorldRouter(asn=3356, name="Level3")
base.addAutonomousSystem(real_as)

# Enable remote access to one of the Autonomous System's network
base.enableRemoteAccess(asn=101, host="customhost", protocol="ssh")

# Set up peering via a route server at IX Frankfurt
ebgp.addRouteServer(ix1.getId())
ebgp.addPeering(asn1=100, asn2=101, ix_id=ix1.getId(), via_route_server=True)
ebgp.addPeering(asn1=100, asn2=102, ix_id=ix1.getId(), via_route_server=True)

# Set up private peering with different peer relationships
ebgp.addPeering(asn1=100, asn2=200, ix_id=ix2.getId(), relationship="transit")
ebgp.addPeering(asn1=101, asn2=103, ix_id=ix3.getId(), relationship="peer")

# Set up iBGP mesh for transit ASes
ibgp.addIbgpMesh(asn=100)
ibgp.addIbgpMesh(asn=200)

# Set up OSPF within stub AS 101
ospf.addOspfArea(asn=101, area=0)

# Add all layers to the emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addLayer(ibgp)
emu.addLayer(ospf)
emu.addLayer(web)

# Save the emulator to a component file
emu.save("emulation_component.bin")

# Render and compile the emulator
emu.render()
emu.compile()
```