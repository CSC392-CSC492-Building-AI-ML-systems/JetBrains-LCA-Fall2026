```python
from seedemu import *

# Create the emulation
emu = Emulation()

# Create Autonomous Systems
as150 = AutonomousSystem(150)
as2 = AutonomousSystem(2)
as151 = AutonomousSystem(151)

# Add routers and networks to AS150
as150_router = Router(1)
as150_network = Network(1)
as150.addRouter(as150_router)
as150.addNetwork(as150_network)
as150_router.addNetwork(as150_network)

# Add routers and networks to AS151
as151_router = Router(1)
as151_network = Network(1)
as151.addRouter(as151_router)
as151.addNetwork(as151_network)
as151_router.addNetwork(as151_network)

# Add routers and networks to AS2
as2_router1 = Router(1)
as2_router2 = Router(2)
as2_network = Network(1)
as2.addRouter(as2_router1)
as2.addRouter(as2_router2)
as2.addNetwork(as2_network)
as2_router1.addNetwork(as2_network)
as2_router2.addNetwork(as2_network)

# Add ASes to the emulation
emu.addAutonomousSystem(as150)
emu.addAutonomousSystem(as2)
emu.addAutonomousSystem(as151)

# Create Internet Exchanges
ix100 = InternetExchange(100)
ix101 = InternetExchange(101)
emu.addInternetExchange(ix100)
emu.addInternetExchange(ix101)

# Connect AS150 and AS2 to IX 100
ix100.addParticipant(as150_router)
ix100.addParticipant(as2_router1)

# Connect AS2 and AS151 to IX 101
ix101.addParticipant(as2_router2)
ix101.addParticipant(as151_router)

# Create BGP attacker component
attacker = BgpAttackerComponent()
attacker.setAsn(65000)
attacker.setName("attacker")
attacker.hijackPrefix(as151_network.getPrefix())
attacker.joinInternetExchange(100)

# Merge attacker component with the emulation
emu.addComponent(attacker)

# Establish private peering relationships
emu.addCrossConnectPeering(150, 2, 100)      # AS150 <-> AS2 at IX 100
emu.addCrossConnectPeering(151, 2, 101)      # AS151 <-> AS2 at IX 101
emu.addCrossConnectPeering(65000, 2, 100)    # Attacker <-> AS2 at IX 100

# Render and compile the simulation with Docker, managing the network internally
emu.render()
emu.compile(Docker(), manageNetworkInternally=True)
```