```python
from seedemu import *

# Create the base emulator
emu = Emulator()

# Create Autonomous Systems
as150 = AutonomousSystem(150)
as2 = AutonomousSystem(2)
as151 = AutonomousSystem(151)

# Add ASes to the emulator
emu.addAutonomousSystem(as150)
emu.addAutonomousSystem(as2)
emu.addAutonomousSystem(as151)

# Create networks for each AS
net150 = Network('net150', '10.150.0.0/24')
net2 = Network('net2', '10.2.0.0/24')
net151 = Network('net151', '10.151.0.0/24')

as150.addNetwork(net150)
as2.addNetwork(net2)
as151.addNetwork(net151)

# Create routers for each AS
router150 = Router('r150')
router2a = Router('r2a')
router2b = Router('r2b')
router151 = Router('r151')

as150.addRouter(router150)
as2.addRouter(router2a)
as2.addRouter(router2b)
as151.addRouter(router151)

# Connect routers to their networks
net150.addInterface(router150.addInterface('if-net150'))
net2.addInterface(router2a.addInterface('if-net2a'))
net2.addInterface(router2b.addInterface('if-net2b'))
net151.addInterface(router151.addInterface('if-net151'))

# Create Internet Exchanges
ix100 = InternetExchange(100, 'ix100', '192.0.2.100/24')
ix101 = InternetExchange(101, 'ix101', '192.0.2.101/24')

emu.addInternetExchange(ix100)
emu.addInternetExchange(ix101)

# Connect routers to IXes
ix100.addInterface(router150.addInterface('if-ix100'))
ix100.addInterface(router2a.addInterface('if-ix100'))
ix101.addInterface(router2b.addInterface('if-ix101'))
ix101.addInterface(router151.addInterface('if-ix101'))

# Create BGP attacker component
attacker = BgpAttackerComponent()
attacker.setAsn(65000)
attacker.setName('attacker')
attacker.setInternetExchange(100)
attacker.setHijackPrefix('10.151.0.0/24')

# Add attacker to the emulator
emu.addComponent(attacker)

# Connect attacker to IX100
attacker_router = Router('attacker-router')
emu.addRouter(attacker_router)
ix100.addInterface(attacker_router.addInterface('if-ix100-attacker'))

# Establish private peerings
emu.addPrivatePeering(150, 2, 100)
emu.addPrivatePeering(151, 2, 101)
emu.addPrivatePeering(65000, 2, 100)

# Render and compile the simulation with Docker, managing the network internally
emu.render()
emu.compile(Docker(), manageNetworkInternally=True)
```