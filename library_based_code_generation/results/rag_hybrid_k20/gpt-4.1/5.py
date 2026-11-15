```python
from seedemu import Emulator
from seedemu.layers import Base, Routing, Service, Bind, InternetExchange, AutonomousSystem, Network, Router, Host, WebService

# Initialize emulator and layers
emulator = Emulator()
base_layer = Base()
routing_layer = Routing()
service_layer = Service()
bind_layer = Bind()

# Create Internet Exchange
ix_id = 100
ix = InternetExchange(ix_id)
ix.setDisplayName("IX-100")
ix.setDescription("Internet Exchange 100 for peering ASes")

# Add Internet Exchange to base layer
base_layer.addInternetExchange(ix)

# Create three Autonomous Systems
as_ids = [200, 201, 202]
as_objs = []
for i, as_id in enumerate(as_ids):
    # Create Autonomous System
    asys = AutonomousSystem(as_id)
    asys.setDisplayName(f"AS-{as_id}")
    asys.setDescription(f"Autonomous System {as_id}")
    base_layer.addAutonomousSystem(asys)
    as_objs.append(asys)

    # Create two networks in the AS
    net1 = Network(f"net{as_id}_1")
    net1.setDisplayName(f"AS{as_id}-Net1")
    net1.setDescription(f"Network 1 in AS {as_id}")
    net2 = Network(f"net{as_id}_2")
    net2.setDisplayName(f"AS{as_id}-Net2")
    net2.setDescription(f"Network 2 in AS {as_id}")
    asys.addNetwork(net1)
    asys.addNetwork(net2)

    # Create router joining both networks
    router = Router(f"r{as_id}")
    router.setDisplayName(f"AS{as_id}-Router")
    router.setDescription(f"Router in AS {as_id}")
    router.joinNetwork(net1)
    router.joinNetwork(net2)
    asys.addRouter(router)

    # Create host joining net1
    host = Host(f"h{as_id}")
    host.setDisplayName(f"AS{as_id}-Host")
    host.setDescription(f"Host in AS {as_id}")
    host.joinNetwork(net1)
    asys.addHost(host)

    # Install web service on a virtual node and bind to host
    web_node = f"web{as_id}"
    service_layer.addService(WebService(web_node))
    bind_layer.bind(web_node, host.getName())

# Peer each AS with the Internet Exchange
for asys in as_objs:
    asys.joinInternetExchange(ix_id)

# Add layers to emulator
emulator.addLayer(base_layer)
emulator.addLayer(routing_layer)
emulator.addLayer(service_layer)
emulator.addLayer(bind_layer)

# Render and compile emulator with Docker, enabling internet map
emulator.render()
emulator.compile(docker=True, internetMap=True)
```