```python
from seed_emulator import Emulator, AutonomousSystem, Router, Network, Host, InternetExchange, BGP

# Create emulator
emu = Emulator()

# Create AS150 (Transit AS)
as150 = AutonomousSystem(asn=150, name="AS150")
routers_150 = [Router(name=f"r150{i+1}") for i in range(4)]
networks_150 = [Network(name=f"net150{i+1}", prefix=f"10.150.{i}.0/24") for i in range(3)]

for router in routers_150:
    as150.add_router(router)
for net, router in zip(networks_150, routers_150[:3]):
    as150.add_network(net)
    router.add_interface(network=net)

# Create AS151
as151 = AutonomousSystem(asn=151, name="AS151")
router_151 = Router(name="r151")
webhost_151 = Host(name="web151")
net151 = Network(name="net151", prefix="10.151.0.0/24")
as151.add_router(router_151)
as151.add_network(net151)
router_151.add_interface(network=net151)
webhost_151.add_interface(network=net151)

# Create AS152
as152 = AutonomousSystem(asn=152, name="AS152")
router_152 = Router(name="r152")
webhost_152 = Host(name="web152")
net152 = Network(name="net152", prefix="10.152.0.0/24")
as152.add_router(router_152)
as152.add_network(net152)
router_152.add_interface(network=net152)
webhost_152.add_interface(network=net152)

# Create Internet Exchange
ix = InternetExchange(name="IX1")

# Connect AS151 and AS152 routers to IX
ix_net = Network(name="ix_net", prefix="192.0.2.0/24")
ix.add_network(ix_net)
router_151.add_interface(network=ix_net)
router_152.add_interface(network=ix_net)

# Connect AS150 routers to AS151 and AS152 routers for BGP peering
# We'll use r1504 for peering
peering_net_151 = Network(name="peering_150_151", prefix="10.200.1.0/30")
peering_net_152 = Network(name="peering_150_152", prefix="10.200.2.0/30")
routers_150[3].add_interface(network=peering_net_151)
router_151.add_interface(network=peering_net_151)
routers_150[3].add_interface(network=peering_net_152)
router_152.add_interface(network=peering_net_152)

# Add BGP peering
bgp_150_151 = BGP(local_as=150, neighbor_as=151, local_router=routers_150[3], neighbor_router=router_151)
bgp_150_152 = BGP(local_as=150, neighbor_as=152, local_router=routers_150[3], neighbor_router=router_152)

# Add all layers to emulator
emu.add_layer(as150)
emu.add_layer(as151)
emu.add_layer(as152)
emu.add_layer(ix)
emu.add_layer(bgp_150_151)
emu.add_layer(bgp_150_152)

# Dump emulator state to binary file
with open("emulator_state.bin", "wb") as f:
    emu.dump(f)
```