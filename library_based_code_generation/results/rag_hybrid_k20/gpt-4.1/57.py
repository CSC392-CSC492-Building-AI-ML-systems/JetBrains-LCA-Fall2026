```python
from seedemu import *

# Create the base emulator
emu = Emulator()

# Create Autonomous Systems
as150 = AutonomousSystem(150)
as2 = AutonomousSystem(2)
as151 = AutonomousSystem(151)

# Create Routers
r150 = Router(1501)
r2a = Router(2001)
r2b = Router(2002)
r151 = Router(1511)

# Add routers to ASes
as150.addRouter(r150)
as2.addRouter(r2a)
as2.addRouter(r2b)
as151.addRouter(r151)

# Create Networks
net150 = Network(15001, "10.150.0.0/24")
net2 = Network(20001, "10.2.0.0/24")
net151 = Network(15101, "10.151.0.0/24")

# Add networks to ASes
as150.addNetwork(net150)
as2.addNetwork(net2)
as151.addNetwork(net151)

# Attach routers to networks
net150.addRouter(r150)
net2.addRouter(r2a)
net151.addRouter(r151)

# Create Internet Exchanges
ix100 = InternetExchange(100, "192.0.2.100/24")
ix101 = InternetExchange(101, "192.0.2.101/24")

# Add IXes to the emulator
emu.addInternetExchange(ix100)
emu.addInternetExchange(ix101)

# Connect routers to IXes
ix100.addRouter(r150)
ix100.addRouter(r2a)
ix101.addRouter(r2b)
ix101.addRouter(r151)

# Add ASes to the emulator
emu.addAutonomousSystem(as150)
emu.addAutonomousSystem(as2)
emu.addAutonomousSystem(as151)

# Create BGP Attacker Component
attacker = BgpAttackerComponent()
attacker.setAsn(65000)
attacker.setRouterId(65001)
attacker.setName("attacker")
attacker.hijackPrefix("10.151.0.0/24")
attacker.joinInternetExchange(100)

# Merge attacker component with the emulator
emu.addComponent(attacker)

# Establish private peerings
emu.addPrivatePeering(150, 2, 100)
emu.addPrivatePeering(151, 2, 101)
emu.addPrivatePeering(65000, 2, 100)

# Render and compile the simulation with Docker, managing the network internally
emu.render()
emu.compile(Docker(), manageNetworkInternally=True)
```