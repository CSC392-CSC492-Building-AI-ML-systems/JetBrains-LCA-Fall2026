```python
from seed_emulator import (
    Emulator,
    BaseLayer,
    RoutingLayer,
    EBGP,
    IBGP,
    OSPF,
    WebServiceLayer,
    AutonomousSystem,
    Router,
    Host,
    Network,
    InternetExchange,
    BGPPeering,
    BGPHijacker,
    merge_emulators,
    DockerCompiler,
)

def create_stub_as(asn, ix, net, prefix, web_port=8080):
    asys = AutonomousSystem(asn=asn)
    router = Router(name=f"r{asn}", asn=asn)
    host = Host(name=f"web{asn}", image="nginx:alpine", ports={web_port: 80})
    asys.add(router)
    asys.add(host)
    asys.add_layer(OSPF())
    asys.add_layer(IBGP())
    asys.add_layer(WebServiceLayer())
    asys.add_layer(RoutingLayer())
    asys.add_layer(BaseLayer())
    asys.add_layer(EBGP())
    net_obj = Network(name=f"net{asn}", prefix=prefix)
    net_obj.add(host)
    net_obj.add(router)
    asys.add(net_obj)
    ix.add(router)
    return asys

# Create emulator and layers
emu = Emulator(name="multi-as-emu")
emu.add_layer(BaseLayer())
emu.add_layer(RoutingLayer())
emu.add_layer(EBGP())
emu.add_layer(IBGP())
emu.add_layer(OSPF())
emu.add_layer(WebServiceLayer())

# Create Internet Exchanges
ix1 = InternetExchange(name="ix1")
ix2 = InternetExchange(name="ix2")
ix3 = InternetExchange(name="ix3")
emu.add(ix1)
emu.add(ix2)
emu.add(ix3)

# Create stub ASes
stub_ases = []
stub_ases.append(create_stub_as(65001, ix1, "net65001", "10.1.0.0/24"))
stub_ases.append(create_stub_as(65002, ix1, "net65002", "10.2.0.0/24"))
stub_ases.append(create_stub_as(65003, ix2, "net65003", "10.3.0.0/24"))
stub_ases.append(create_stub_as(65004, ix2, "net65004", "10.4.0.0/24"))
stub_ases.append(create_stub_as(65005, ix3, "net65005", "10.5.0.0/24"))
stub_ases.append(create_stub_as(65006, ix3, "net65006", "10.6.0.0/24"))

for asys in stub_ases:
    emu.add(asys)

# Create two main ASes with routers joining different networks and IXs
as100 = AutonomousSystem(asn=100)
r100a = Router(name="r100a", asn=100)
r100b = Router(name="r100b", asn=100)
net100a = Network(name="net100a", prefix="192.168.100.0/24")
net100b = Network(name="net100b", prefix="192.168.101.0/24")
net100a.add(r100a)
net100b.add(r100b)
as100.add(r100a)
as100.add(r100b)
as100.add(net100a)
as100.add(net100b)
as100.add_layer(OSPF())
as100.add_layer(IBGP())
as100.add_layer(RoutingLayer())
as100.add_layer(BaseLayer())
as100.add_layer(EBGP())
ix1.add(r100a)
ix2.add(r100b)
emu.add(as100)

as200 = AutonomousSystem(asn=200)
r200a = Router(name="r200a", asn=200)
r200b = Router(name="r200b", asn=200)
net200a = Network(name="net200a", prefix="192.168.200.0/24")
net200b = Network(name="net200b", prefix="192.168.201.0/24")
net200a.add(r200a)
net200b.add(r200b)
as200.add(r200a)
as200.add(r200b)
as200.add(net200a)
as200.add(net200b)
as200.add_layer(OSPF())
as200.add_layer(IBGP())
as200.add_layer(RoutingLayer())
as200.add_layer(BaseLayer())
as200.add_layer(EBGP())
ix2.add(r200a)
ix3.add(r200b)
emu.add(as200)

# Define private peerings between different ASes
emu.add(BGPPeering(asn1=100, router1="r100a", asn2=200, router2="r200a", type="private"))
emu.add(BGPPeering(asn1=100, router1="r100b", asn2=65003, router2="r65003", type="private"))
emu.add(BGPPeering(asn1=200, router1="r200b", asn2=65006, router2="r65006", type="private"))

# Add BGP attacker component
attacker = Emulator(name="bgp-attacker")
attacker_asn = 666
attacker_as = AutonomousSystem(asn=attacker_asn)
attacker_router = Router(name="r666", asn=attacker_asn)
ix1.add(attacker_router)
attacker_as.add(attacker_router)
attacker_as.add_layer(EBGP())
attacker_as.add_layer(BaseLayer())
# Hijack prefixes from stub ASes
hijacker = BGPHijacker(router=attacker_router, prefixes=["10.1.0.0/24", "10.2.0.0/24"])
attacker.add(attacker_as)
attacker.add(hijacker)

# Merge attacker with main emulator
emu = merge_emulators(emu, attacker)

# Render and compile
emu.render()
compiler = DockerCompiler(emu)
compiler.compile(output_dir="./output_emulator")
```