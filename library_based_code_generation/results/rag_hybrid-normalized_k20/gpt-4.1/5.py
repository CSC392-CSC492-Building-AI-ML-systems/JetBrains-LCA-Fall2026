```python
from seedemu import Emulator
from seedemu.layers import Base, Routing, Service, Bind
from seedemu.core import InternetExchange, AutonomousSystem

# Initialize emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
service = Service()
bind = Bind()

# Create Internet Exchange
ix_id = 100
ix = InternetExchange(ix_id)
ix.setDisplayName("IX-100")
ix.setDescription("Internet Exchange 100 for peering ASes")
base.addInternetExchange(ix)

# Create three Autonomous Systems
as_ids = [200, 201, 202]
for as_id in as_ids:
    # Create AS
    asys = AutonomousSystem(as_id)
    asys.setDisplayName(f"AS-{as_id}")
    asys.setDescription(f"Autonomous System {as_id}")
    base.addAutonomousSystem(asys)
    
    # Create two networks in the AS
    net1 = base.createNetwork(as_id)
    net1.setDisplayName(f"AS{as_id}-Net1")
    net1.setDescription(f"Network 1 in AS {as_id}")
    net2 = base.createNetwork(as_id)
    net2.setDisplayName(f"AS{as_id}-Net2")
    net2.setDescription(f"Network 2 in AS {as_id}")
    
    # Create router joining both networks
    router = base.createRouter(as_id)
    router.setDisplayName(f"AS{as_id}-Router")
    router.setDescription(f"Router in AS {as_id}")
    base.joinNetwork(router, net1)
    base.joinNetwork(router, net2)
    
    # Create host joining net1
    host = base.createHost(as_id)
    host.setDisplayName(f"AS{as_id}-Host")
    host.setDescription(f"Host in AS {as_id}")
    base.joinNetwork(host, net1)
    
    # Create virtual node for web service
    vnode = service.createVirtualNode()
    service.installWebService(vnode)
    bind.bind(vnode, host)
    
    # Peer AS with Internet Exchange
    base.joinInternetExchange(router, ix_id)

# Add layers to emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(service)
emu.addLayer(bind)

# Render and compile emulator with Docker and internet map enabled
emu.render()
emu.compile(docker=True, internetMap=True)
```