```python
from seedemu import Emulator
from seedemu.layers import Base, Routing, Service, InternetExchange, Bind
from seedemu.core import AutonomousSystem, Network, Router, Host, VirtualNode

# Initialize emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
service = Service()
ix_layer = InternetExchange()
bind = Bind()

# Create Internet Exchange
ix_id = 100
ix = ix_layer.createInternetExchange(ix_id)
ix.setDisplayName("IX-100")
ix.setDescription("Internet Exchange with ID 100")

# Create three Autonomous Systems
as_ids = [200, 201, 202]
for i, asn in enumerate(as_ids):
    # Create AS
    asys = base.createAutonomousSystem(asn)
    asys.setDisplayName(f"AS-{asn}")
    asys.setDescription(f"Autonomous System {asn}")

    # Create two networks in the AS
    net1 = base.createNetwork(f"10.{i}.0.0/24", asn)
    net1.setDisplayName(f"AS{asn}-Net1")
    net1.setDescription(f"First network in AS {asn}")

    net2 = base.createNetwork(f"10.{i}.1.0/24", asn)
    net2.setDisplayName(f"AS{asn}-Net2")
    net2.setDescription(f"Second network in AS {asn}")

    # Create router joining both networks
    router = base.createRouter(asn)
    router.joinNetwork(net1)
    router.joinNetwork(net2)
    router.setDisplayName(f"AS{asn}-Router")
    router.setDescription(f"Router for AS {asn}")

    # Create host joining net1
    host = base.createHost(asn)
    host.joinNetwork(net1)
    host.setDisplayName(f"AS{asn}-Host")
    host.setDescription(f"Host in AS {asn}")

    # Create virtual node, install web service, bind to host
    vnode = VirtualNode()
    vnode.setDisplayName(f"AS{asn}-WebNode")
    vnode.setDescription(f"Web service node in AS {asn}")
    service.installWebService(vnode)
    bind.bind(vnode, host)

    # Peer router with IX
    ix_layer.joinInternetExchange(router, ix_id)

# Add layers to emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(service)
emu.addLayer(ix_layer)
emu.addLayer(bind)

# Render and compile emulator with Docker and internet map
emu.render()
emu.compile(docker=True, internetMap=True)
```