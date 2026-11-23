```python
from seed_emulator import (
    createEmulation,
    makeStubAsWithHosts,
    WebService,
    getNodeByAsnAndName,
    gen_emulation_files,
    joinInternetExchange,
    makeTransitAs,
)

# Create the emulation
emu = createEmulation()

# Create AS150, AS151, AS152 with 'web' host and 'router0'
as150 = makeStubAsWithHosts(150, hosts=['web'], routers=['router0'])
as151 = makeStubAsWithHosts(151, hosts=['web'], routers=['router0'])
as152 = makeStubAsWithHosts(152, hosts=['web'], routers=['router0'])

emu.add_as(as150)
emu.add_as(as151)
emu.add_as(as152)

# Add 'net0' to each AS and connect 'web' and 'router0' to it
for asys in [as150, as151, as152]:
    net0 = asys.add_network('net0')
    asys.get_host('web').join_network(net0)
    asys.get_router('router0').join_network(net0)

# Install web service on each 'web' host
for asys in [as150, as151, as152]:
    web_host = asys.get_host('web')
    web_host.install_service(WebService())

# Cross connection between AS150 and AS152 routers
router150 = as150.get_router('router0')
router152 = as152.get_router('router0')
emu.add_cross_connection(router150, router152)

# Create Internet Exchange 100 and join AS150 and AS151 as peers
ix100 = emu.create_internet_exchange(100)
joinInternetExchange(as150, ix100)
joinInternetExchange(as151, ix100)

# AS150 is a provider for AS152
as150.add_customer(as152)

# Render and compile emulation using Docker with self-managed network
gen_emulation_files(
    emu,
    output_dir='./cross-connect',
    render=True,
    compile=True,
    docker=True,
    self_managed_network=True
)
```