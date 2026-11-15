```python
from seed_emulator import (
    createEmulation,
    createInternetExchange,
    createAutonomousSystem,
    joinInternetExchange,
    setInternetExchange,
    setAutonomousSystem,
    Emulator,
    build_emulator,
    gen_emulation_files,
    BgpAttackerComponent,
)

def create_stub_as_with_web(asn, ix, net_name):
    # Create stub AS
    stub_as = createAutonomousSystem(
        asn=asn,
        layers=["base", "routing", "eBGP", "iBGP", "OSPF", "web_service"]
    )
    # Add router and web server
    router = stub_as.add_router(f"r{asn}")
    web = stub_as.add_host(f"web{asn}")
    # Connect router and web server to a network
    net = stub_as.add_network(net_name)
    net.add_node(router)
    net.add_node(web)
    # Join the router to the internet exchange
    joinInternetExchange(router, ix)
    return stub_as

# Create the base emulation
emu = createEmulation(layers=["base", "routing", "eBGP", "iBGP", "OSPF", "web_service"])

# Create three internet exchanges
ix1 = createInternetExchange("IX1")
ix2 = createInternetExchange("IX2")
ix3 = createInternetExchange("IX3")
emu.add_internet_exchange(ix1)
emu.add_internet_exchange(ix2)
emu.add_internet_exchange(ix3)

# Create multiple stub ASes and join them to exchanges
stub_ases = []
for i in range(10, 13):
    stub_as = create_stub_as_with_web(asn=i, ix=ix1, net_name=f"net{i}")
    emu.add_autonomous_system(stub_as)
    stub_ases.append(stub_as)

for i in range(13, 16):
    stub_as = create_stub_as_with_web(asn=i, ix=ix2, net_name=f"net{i}")
    emu.add_autonomous_system(stub_as)
    stub_ases.append(stub_as)

for i in range(16, 19):
    stub_as = create_stub_as_with_web(asn=i, ix=ix3, net_name=f"net{i}")
    emu.add_autonomous_system(stub_as)
    stub_ases.append(stub_as)

# Create two autonomous systems with routers joining different networks and exchanges
as100 = createAutonomousSystem(
    asn=100,
    layers=["base", "routing", "eBGP", "iBGP", "OSPF", "web_service"]
)
r100a = as100.add_router("r100a")
r100b = as100.add_router("r100b")
net100a = as100.add_network("net100a")
net100b = as100.add_network("net100b")
net100a.add_node(r100a)
net100b.add_node(r100b)
joinInternetExchange(r100a, ix1)
joinInternetExchange(r100b, ix2)
emu.add_autonomous_system(as100)

as200 = createAutonomousSystem(
    asn=200,
    layers=["base", "routing", "eBGP", "iBGP", "OSPF", "web_service"]
)
r200a = as200.add_router("r200a")
r200b = as200.add_router("r200b")
net200a = as200.add_network("net200a")
net200b = as200.add_network("net200b")
net200a.add_node(r200a)
net200b.add_node(r200b)
joinInternetExchange(r200a, ix2)
joinInternetExchange(r200b, ix3)
emu.add_autonomous_system(as200)

# Define private peerings between different autonomous systems
as100.add_private_peering(r100a, r100b)
as200.add_private_peering(r200a, r200b)
# Private peering between as100 and as200
as100.add_private_peering(r100a, r200a)
as100.add_private_peering(r100b, r200b)

# Add a BGP attacker component that hijacks certain prefixes and joins an internet exchange
bgp_attacker = BgpAttackerComponent(
    asn=666,
    hijack_prefixes=["10.0.0.0/24", "192.168.1.0/24"],
    layers=["base", "routing", "eBGP"]
)
attacker_router = bgp_attacker.add_router("attacker-router")
joinInternetExchange(attacker_router, ix1)

# Merge the BGP attacker with the emulator
emu.add_autonomous_system(bgp_attacker)

# Render the new emulator
emu.render()

# Compile the new emulator using Docker and output to a specified directory
output_dir = "./emulator_output"
build_emulator(emu, output_dir=output_dir, use_docker=True)
```