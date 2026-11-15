```python
from seedemu import *
from seedemu.layers import *
from seedemu.components import *

# Create the base emulator
emu = Emulator()

# Create the network layer
network = Network()

# AS150: one router, one network
network.createAutonomousSystem(150)
network.createRouter(150, 'r1')
network.createNetwork(150, 'net150', '10.150.0.0/24')
network.joinNetwork(150, 'r1', 'net150')

# AS151: one router, one network
network.createAutonomousSystem(151)
network.createRouter(151, 'r1')
network.createNetwork(151, 'net151', '10.151.0.0/24')
network.joinNetwork(151, 'r1', 'net151')

# AS2: two routers, one network
network.createAutonomousSystem(2)
network.createRouter(2, 'r1')
network.createRouter(2, 'r2')
network.createNetwork(2, 'net2', '10.2.0.0/24')
network.joinNetwork(2, 'r1', 'net2')
network.joinNetwork(2, 'r2', 'net2')

# Add the network layer to the emulator
emu.addLayer(network)

# Create the internet exchange layer
ix = InternetExchange()

# IX 100: connect AS150 and AS2
ix.createInternetExchange(100, 'ix100')
ix.joinInternetExchange(100, 150, 'r1')
ix.joinInternetExchange(100, 2, 'r1')

# IX 101: connect AS2 and AS151
ix.createInternetExchange(101, 'ix101')
ix.joinInternetExchange(101, 2, 'r2')
ix.joinInternetExchange(101, 151, 'r1')

emu.addLayer(ix)

# Create the BGP layer
bgp = Bgp()

# Add BGP routers for AS150, AS151, AS2
bgp.addRouter(150, 'r1')
bgp.addRouter(151, 'r1')
bgp.addRouter(2, 'r1')
bgp.addRouter(2, 'r2')

# Add BGP peering at IX 100 between AS150 and AS2
bgp.addPeer(150, 'r1', 2, 'r1', 100, BgpPeeringType.Private)
bgp.addPeer(2, 'r1', 150, 'r1', 100, BgpPeeringType.Private)

# Add BGP peering at IX 101 between AS151 and AS2
bgp.addPeer(151, 'r1', 2, 'r2', 101, BgpPeeringType.Private)
bgp.addPeer(2, 'r2', 151, 'r1', 101, BgpPeeringType.Private)

emu.addLayer(bgp)

# Create the BGP attacker component
attacker = BgpAttackerComponent(asn=666, routerName='attacker', prefix='10.151.0.0/24')
attacker.joinInternetExchange(100)

# Add attacker to the emulator
emu.addComponent(attacker)

# Add attacker BGP peering with AS2 at IX 100
bgp.addRouter(666, 'attacker')
bgp.addPeer(666, 'attacker', 2, 'r1', 100, BgpPeeringType.Private)
bgp.addPeer(2, 'r1', 666, 'attacker', 100, BgpPeeringType.Private)

# Render and compile the simulation with Docker, managing the network internally
emu.render()
emu.compile(Docker(), manageNetwork=True)
```