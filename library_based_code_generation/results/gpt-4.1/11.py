```python
from seed_emulator import Emulator, Layer, AutonomousSystem, Router, Host, Network, eBGP

# Base Layer
base_layer = Layer("base")

# AS1: 5 hosts, 1 router, all join one network
as1 = AutonomousSystem("AS1")
as1_net = Network("10.1.0.0/24")
as1_router = Router("as1-r1")
as1.add_router(as1_router)
as1_net.add_device(as1_router)
for i in range(1, 6):
    host = Host(f"as1-h{i}")
    as1.add_host(host)
    as1_net.add_device(host)
as1.add_network(as1_net)
base_layer.add_autonomous_system(as1)

# AS2: 3 routers, each joins a different network
as2 = AutonomousSystem("AS2")
for i in range(1, 4):
    net = Network(f"10.2.{i}.0/24")
    router = Router(f"as2-r{i}")
    as2.add_router(router)
    net.add_device(router)
    as2.add_network(net)
base_layer.add_autonomous_system(as2)

# AS3: 2 routers, both join the same network
as3 = AutonomousSystem("AS3")
as3_net = Network("10.3.0.0/24")
for i in range(1, 3):
    router = Router(f"as3-r{i}")
    as3.add_router(router)
    as3_net.add_device(router)
as3.add_network(as3_net)
base_layer.add_autonomous_system(as3)

# Routing Layer
routing_layer = Layer("routing")
# (Assume routing configuration is handled by the library defaults or omitted for brevity)

# eBGP Layer
ebgp_layer = Layer("eBGP")
# Add private peering between different ASes
ebgp_layer.add_ebgp_peering(as1.get_router("as1-r1"), as2.get_router("as2-r1"))
ebgp_layer.add_ebgp_peering(as2.get_router("as2-r2"), as3.get_router("as3-r1"))
ebgp_layer.add_ebgp_peering(as1.get_router("as1-r1"), as3.get_router("as3-r2"))

# Emulator
emulator = Emulator()
emulator.add_layer(base_layer)
emulator.add_layer(routing_layer)
emulator.add_layer(ebgp_layer)

# Dump emulator state to binary file
with open("emulator_state.bin", "wb") as f:
    emulator.dump(f)
```