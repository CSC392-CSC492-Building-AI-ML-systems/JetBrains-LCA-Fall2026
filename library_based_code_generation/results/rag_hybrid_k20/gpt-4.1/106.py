```python
from seed_emulator import (
    createEmulation,
    createInternetExchange,
    createAutonomousSystem,
    joinInternetExchange,
    setInternetExchange,
    Emulator,
    build_emulator,
    gen_emulation_files,
    BgpAttackerComponent,
)
from seed_emulator.layers import (
    BaseLayer,
    RoutingLayer,
    EBGP,
    IBGP,
    OSPF,
    WebServiceLayer,
)
import os

def create_stub_as_with_web(asn, ix, net_prefix):
    # Create stub AS
    stub_as = createAutonomousSystem(asn=asn)
    # Add router
    router = stub_as.add_router(f"r{asn}")
    # Add web server
    web = stub_as.add_host(f"web{asn}")
    # Connect router and web server to the network
    net = stub_as.add_network(f"net{asn}", prefix=net_prefix)
    net.add_interface(router)
    net.add_interface(web)
    # Join the internet exchange
    joinInternetExchange(stub_as, ix)
    return stub_as

# Create base emulation with required layers
emulation = createEmulation(
    layers=[
        BaseLayer(),
        RoutingLayer(),
        EBGP(),
        IBGP(),
        OSPF(),
        WebServiceLayer(),
    ]
)

# Create three internet exchanges
ix1 = createInternetExchange("ix1", prefix="10.100.1.0/24")
ix2 = createInternetExchange("ix2", prefix="10.100.2.0/24")
ix3 = createInternetExchange("ix3", prefix="10.100.3.0/24")

emulation.add_internet_exchange(ix1)
emulation.add_internet_exchange(ix2)
emulation.add_internet_exchange(ix3)

# Create multiple stub ASes and join exchanges
stub_ases = []
for i in range(10, 13):
    stub_as = create_stub_as_with_web(asn=i, ix=ix1, net_prefix=f"10.{i}.0.0/24")
    emulation.add_autonomous_system(stub_as)
    stub_ases.append(stub_as)

for i in range(13, 16):
    stub_as = create_stub_as_with_web(asn=i, ix=ix2, net_prefix=f"10.{i}.0.0/24")
    emulation.add_autonomous_system(stub_as)
    stub_ases.append(stub_as)

for i in range(16, 19):
    stub_as = create_stub_as_with_web(asn=i, ix=ix3, net_prefix=f"10.{i}.0.0/24")
    emulation.add_autonomous_system(stub_as)
    stub_ases.append(stub_as)

# Create two autonomous systems with routers joining different networks and exchanges
as20 = createAutonomousSystem(asn=20)
r20a = as20.add_router("r20a")
r20b = as20.add_router("r20b")
net20a = as20.add_network("net20a", prefix="10.20.1.0/24")
net20b = as20.add_network("net20b", prefix="10.20.2.0/24")
net20a.add_interface(r20a)
net20b.add_interface(r20b)
joinInternetExchange(as20, ix1)
joinInternetExchange(as20, ix2)
emulation.add_autonomous_system(as20)

as30 = createAutonomousSystem(asn=30)
r30a = as30.add_router("r30a")
r30b = as30.add_router("r30b")
net30a = as30.add_network("net30a", prefix="10.30.1.0/24")
net30b = as30.add_network("net30b", prefix="10.30.2.0/24")
net30a.add_interface(r30a)
net30b.add_interface(r30b)
joinInternetExchange(as30, ix2)
joinInternetExchange(as30, ix3)
emulation.add_autonomous_system(as30)

# Define private peerings between different autonomous systems
as20.add_private_peering(as30)
as30.add_private_peering(as20)
for stub_as in stub_ases:
    as20.add_private_peering(stub_as)
    stub_as.add_private_peering(as20)

# Add a BGP attacker component that hijacks certain prefixes and joins an internet exchange
attacker = BgpAttackerComponent(
    asn=65000,
    hijack_prefixes=["10.13.0.0/24", "10.16.0.0/24"],
    name="bgp_attacker"
)
joinInternetExchange(attacker, ix2)

# Merge the BGP attacker with the emulator and render the new emulator
emulator = Emulator(emulation)
emulator.add_component(attacker)
emulator.render()

# Compile the new emulator using Docker and output the result to a specified directory
output_dir = "./emulator_output"
os.makedirs(output_dir, exist_ok=True)
build_emulator(emulator, output_dir=output_dir, use_docker=True)
```