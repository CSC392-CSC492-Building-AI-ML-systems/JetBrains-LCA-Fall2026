```python
from seed_emulator import Emulation, AutonomousSystem, Host, Router, Network, InternetExchange

# Create the emulation
emu = Emulation()

# Create ASes
ases = {}
for asn in [150, 151, 152]:
    asys = AutonomousSystem(asn)
    # Create network
    net = Network('net0')
    asys.add_network(net)
    # Create router
    router = Router('router0')
    asys.add_router(router)
    net.add_member(router)
    # Create web host
    web = Host('web')
    web.install_service('web')
    asys.add_host(web)
    net.add_member(web)
    # Add AS to emulation and dict
    emu.add_as(asys)
    ases[asn] = {'as': asys, 'router': router}

# Cross connection between AS150 and AS152 routers
emu.add_cross_connection(ases[150]['router'], ases[152]['router'])

# Create Internet Exchange 100
ix = InternetExchange(100)
emu.add_internet_exchange(ix)

# AS150 and AS151 are peers on IX100
ix.add_peer(ases[150]['as'])
ix.add_peer(ases[151]['as'])

# AS150 is provider for AS152
ases[150]['as'].add_customer(ases[152]['as'])

# Render and compile with Docker, self-managed network
emu.render()
emu.compile(
    backend='docker',
    output_dir='./cross-connect',
    network_mode='self-managed'
)
```