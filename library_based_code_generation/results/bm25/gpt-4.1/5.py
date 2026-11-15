```python
from seedemu import Emulator
from seedemu.layers import Base, Routing, Service, Bind, InternetMap
from seedemu.core import AutonomousSystem, InternetExchange

# Initialize emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
service = Service()
bind = Bind()
internet_map = InternetMap()

# Create Internet Exchange
ix_id = 100
ix = InternetExchange(ix_id)
ix.setDisplayName("IX-100")
ix.setDescription("Internet Exchange 100 for peering ASes")
base.addInternetExchange(ix)

# Create three Autonomous Systems
as_ids = [200, 201, 202]
for i, asn in enumerate(as_ids):
    # Create AS
    asys = AutonomousSystem(asn)
    asys.setDisplayName(f"AS-{asn}")
    asys.setDescription(f"Autonomous System {asn}")
    base.addAutonomousSystem(asys)
    
    # Create two networks in AS
    net1 = base.createNetwork(asn, f"net{asn}_1")
    net1.setDisplayName(f"AS{asn} Network 1")
    net1.setDescription(f"First network in AS{asn}")
    net2 = base.createNetwork(asn, f"net{asn}_2")
    net2.setDisplayName(f"AS{asn} Network 2")
    net2.setDescription(f"Second network in AS{asn}")
    
    # Create router joining both networks
    router = base.createRouter(asn, f"rtr{asn}")
    router.setDisplayName(f"AS{asn} Router")
    router.setDescription(f"Router for AS{asn} joining two networks")
    base.joinNetwork(asn, router.getName(), net1.getName())
    base.joinNetwork(asn, router.getName(), net2.getName())
    
    # Create host joining net1
    host = base.createHost(asn, f"host{asn}")
    host.setDisplayName(f"AS{asn} Host")
    host.setDescription(f"Host in AS{asn} on Network 1")
    base.joinNetwork(asn, host.getName(), net1.getName())
    
    # Create virtual node and install web service
    vweb = service.createVirtualNode(asn, f"web{asn}")
    service.installPoAServer(asn, vweb.getName())
    
    # Bind virtual node to host
    bind.bindVirtualNode(asn, vweb.getName(), host.getName())
    
    # Peer AS with Internet Exchange
    base.joinInternetExchange(asn, ix_id, router.getName())

# Add layers to emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(service)
emu.addLayer(bind)
emu.addLayer(internet_map)

# Render and compile emulator with Docker and internet map enabled
emu.render()
emu.compile(docker=True, internetMap=True)
```