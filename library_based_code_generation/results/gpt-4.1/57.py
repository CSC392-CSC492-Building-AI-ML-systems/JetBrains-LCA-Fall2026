```python
from seedemu import *
from seedemu.layers import *
from seedemu.components import *

# Create the base emulator
emu = Emulator()

# Create the network layer
net = Network()

# AS150: 1 router, 1 network
net.addRouter(asn=150, id=1)
net.addNetwork(asn=150, id=1, prefix="10.150.0.0/24")
net.addRouterToNetwork(asn=150, router=1, network=1)

# AS151: 1 router, 1 network
net.addRouter(asn=151, id=1)
net.addNetwork(asn=151, id=1, prefix="10.151.0.0/24")
net.addRouterToNetwork(asn=151, router=1, network=1)

# AS2: 2 routers, 1 network
net.addRouter(asn=2, id=1)
net.addRouter(asn=2, id=2)
net.addNetwork(asn=2, id=1, prefix="10.2.0.0/24")
net.addRouterToNetwork(asn=2, router=1, network=1)
net.addRouterToNetwork(asn=2, router=2, network=1)

emu.addLayer(net)

# Create the internet exchange layer
ix = InternetExchange()

# IX 100: connect AS150 and AS2
ix.addIx(id=100)
ix.addParticipant(asn=150, ix=100, router=1)
ix.addParticipant(asn=2, ix=100, router=1)

# IX 101: connect AS2 and AS151
ix.addIx(id=101)
ix.addParticipant(asn=2, ix=101, router=2)
ix.addParticipant(asn=151, ix=101, router=1)

emu.addLayer(ix)

# Create the BGP layer
bgp = Bgp()

# Add BGP routers for AS150, AS151, AS2
bgp.addRouter(asn=150, id=1)
bgp.addRouter(asn=151, id=1)
bgp.addRouter(asn=2, id=1)
bgp.addRouter(asn=2, id=2)

# Add BGP prefixes
bgp.addPrefix(asn=150, prefix="10.150.0.0/24")
bgp.addPrefix(asn=151, prefix="10.151.0.0/24")
bgp.addPrefix(asn=2, prefix="10.2.0.0/24")

# Add BGP sessions for IXes (will be completed after attacker is added)
emu.addLayer(bgp)

# Create the attacker component
attacker = BgpAttacker(asn=666, routerId=1)
attacker.addIx(ix=100)
attacker.hijack(prefix="10.151.0.0/24")

# Merge attacker component
emu.addComponent(attacker)

# Add attacker to network and IX
net.addRouter(asn=666, id=1)
ix.addParticipant(asn=666, ix=100, router=1)
bgp.addRouter(asn=666, id=1)

# Establish private peering relationships
# AS150 <-> AS2 at IX 100
bgp.addPrivatePeering(asn1=150, router1=1, asn2=2, router2=1, ix=100)
# AS151 <-> AS2 at IX 101
bgp.addPrivatePeering(asn1=151, router1=1, asn2=2, router2=2, ix=101)
# Attacker <-> AS2 at IX 100
bgp.addPrivatePeering(asn1=666, router1=1, asn2=2, router2=1, ix=100)

# Render and compile the simulation with Docker, managing the network internally
emu.render()
emu.compile(Docker(), manageNetwork=True)
```