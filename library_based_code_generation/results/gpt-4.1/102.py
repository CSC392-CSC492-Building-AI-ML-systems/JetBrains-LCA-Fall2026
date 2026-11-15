```python
from seedemu import Emulator
from seedemu.layers import Base, Routing, Ebgp
from seedemu.core import Network, Host, Router, AutonomousSystem, InternetExchange

def create_stub_as(base_layer, routing_layer, asn, exchange, num_hosts=2):
    # Create AS
    asys = AutonomousSystem(asn)
    base_layer.addAutonomousSystem(asys)
    
    # Create network for the AS
    net = Network(f"net{asn}")
    base_layer.addNetwork(net)
    
    # Create hosts and add to network and AS
    hosts = []
    for i in range(1, num_hosts + 1):
        host = Host(f"as{asn}-h{i}")
        base_layer.addHost(host)
        base_layer.joinNetwork(host.getName(), net.getName())
        base_layer.joinAutonomousSystem(host.getName(), asn)
        hosts.append(host)
    
    # Create router, add to network and AS
    router = Router(f"as{asn}-r")
    base_layer.addRouter(router)
    base_layer.joinNetwork(router.getName(), net.getName())
    base_layer.joinAutonomousSystem(router.getName(), asn)
    
    # Router joins the specified exchange
    base_layer.joinInternetExchange(router.getName(), exchange.getName())
    
    # Add router to routing layer
    routing_layer.addRouter(asn, router.getName())
    
    return {
        'asys': asys,
        'hosts': hosts,
        'router': router,
        'network': net
    }

# Create emulator and layers
emu = Emulator()
base = Base()
routing = Routing()
ebgp = Ebgp()

# Create Internet Exchanges
ix1 = InternetExchange("ix1")
ix2 = InternetExchange("ix2")
base.addInternetExchange(ix1)
base.addInternetExchange(ix2)

# Create multiple stub ASes
as1 = create_stub_as(base, routing, 65001, ix1)
as2 = create_stub_as(base, routing, 65002, ix1)
as3 = create_stub_as(base, routing, 65003, ix2)
as4 = create_stub_as(base, routing, 65004, ix2)

# Add private peering between AS1 and AS3, and between AS2 and AS4
ebgp.addPrivatePeering(65001, as1['router'].getName(), 65003, as3['router'].getName())
ebgp.addPrivatePeering(65002, as2['router'].getName(), 65004, as4['router'].getName())

# Add layers to emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)

# Dump emulator state to binary file
emu.dump("emulation.bin")
```