```python
from seed_emulator import (
    createEmulation,
    createInternetExchange,
    createAutonomousSystem,
    joinInternetExchange,
    joinNetwork,
    getInternetExchange,
    getNodeByAsnAndName,
    BgpAttackerComponent,
    shouldMerge,
)
from seed_emulator.layers import (
    base,
    routing,
    eBGP,
    iBGP,
    OSPF,
    web_service,
)
import os

def create_stub_as_with_web(asn, ix, network):
    # Create stub AS
    stub_as = createAutonomousSystem(asn=asn)
    # Add router and web server
    router = stub_as.addRouter(f"r{asn}")
    web = stub_as.addHost(f"web{asn}")
    # Connect web server to router
    stub_as.addLink(router, web)
    # Join network and IX
    joinNetwork(router, network)
    joinInternetExchange(router, ix)
    return stub_as

# Create base emulation with required layers
emu = createEmulation(
    layers=[base, routing, eBGP, iBGP, OSPF, web_service]
)

# Create three internet exchanges
ix1 = createInternetExchange("IX1")
ix2 = createInternetExchange("IX2")
ix3 = createInternetExchange("IX3")
emu.addInternetExchange(ix1)
emu.addInternetExchange(ix2)
emu.addInternetExchange(ix3)

# Create networks for stub ASes
networks = [f"10.{i}.0.0/24" for i in range(1, 10)]

# Create multiple stub ASes and join them to IXes
stub_ases = []
for i, (asn, ix, net) in enumerate([
    (65001, ix1, networks[0]),
    (65002, ix1, networks[1]),
    (65003, ix2, networks[2]),
    (65004, ix2, networks[3]),
    (65005, ix3, networks[4]),
    (65006, ix3, networks[5]),
]):
    stub_as = create_stub_as_with_web(asn, ix, net)
    emu.addAutonomousSystem(stub_as)
    stub_ases.append(stub_as)

# Create two non-stub ASes with routers joining different networks and IXes
as100 = createAutonomousSystem(asn=100)
r100a = as100.addRouter("r100a")
r100b = as100.addRouter("r100b")
joinNetwork(r100a, networks[6])
joinInternetExchange(r100a, ix1)
joinInternetExchange(r100b, ix2)
emu.addAutonomousSystem(as100)

as200 = createAutonomousSystem(asn=200)
r200a = as200.addRouter("r200a")
r200b = as200.addRouter("r200b")
joinNetwork(r200a, networks[7])
joinInternetExchange(r200a, ix2)
joinInternetExchange(r200b, ix3)
emu.addAutonomousSystem(as200)

# Define private peerings between different ASes
as100.addPrivatePeering(r100a, r200a)
as200.addPrivatePeering(r200b, r100b)

# Add BGP attacker component that hijacks certain prefixes and joins an IX
attacker = BgpAttackerComponent(
    asn=666,
    hijack_prefixes=["10.1.0.0/24", "10.2.0.0/24"],
    name="bgp_attacker"
)
attacker_router = attacker.addRouter("attacker-router")
joinInternetExchange(attacker_router, ix1)

# Merge attacker with emulator
if shouldMerge(emu, attacker):
    emu = emu.merge(attacker)

# Render the new emulator
emu.render()

# Compile the emulator using Docker and output to specified directory
output_dir = "./emulator_output"
os.makedirs(output_dir, exist_ok=True)
emu.compile(backend="docker", output_dir=output_dir)
```