```python
from seedemu import Emulator, createInternetExchange, createAutonomousSystem, createNetwork

# Initialize emulator and layers
emu = Emulator()

# Create Internet Exchange
ix = createInternetExchange(100)
ix.setDisplayName("IX-100")
ix.setDescription("Main Internet Exchange for demonstration.")

# Create three Autonomous Systems
as_list = []
for as_id in [200, 201, 202]:
    asys = createAutonomousSystem(as_id)
    asys.setDisplayName(f"AS-{as_id}")
    asys.setDescription(f"Autonomous System {as_id} for testing.")
    
    # Create two networks per AS
    net1 = createNetwork(f"net{as_id}a")
    net1.setDisplayName(f"AS{as_id}-NetA")
    net1.setDescription(f"Network A in AS {as_id}")
    net2 = createNetwork(f"net{as_id}b")
    net2.setDisplayName(f"AS{as_id}-NetB")
    net2.setDescription(f"Network B in AS {as_id}")
    
    # Add networks to AS
    asys.addNetwork(net1)
    asys.addNetwork(net2)
    
    # Create router joining both networks
    router = asys.createRouter(f"r{as_id}")
    router.joinNetwork(net1)
    router.joinNetwork(net2)
    router.setDisplayName(f"AS{as_id}-Router")
    router.setDescription(f"Router in AS {as_id} joining NetA and NetB")
    
    # Create host joining net1
    host = asys.createHost(f"h{as_id}")
    host.joinNetwork(net1)
    host.setDisplayName(f"AS{as_id}-Host")
    host.setDescription(f"Host in AS {as_id} on NetA")
    
    # Create virtual node, install web service, bind to host
    vnode = asys.createVirtualNode(f"vweb{as_id}")
    vnode.installWebService()
    vnode.bindToHost(host)
    
    as_list.append(asys)

# Add Internet Exchange and ASes to emulator
emu.addLayer(ix)
for asys in as_list:
    emu.addLayer(asys)

# Peer each AS with the Internet Exchange
for asys in as_list:
    asys.peerWithInternetExchange(ix)

# Render and compile emulator with Docker and internet map enabled
emu.render()
emu.compile(docker=True, internet_map=True)
```