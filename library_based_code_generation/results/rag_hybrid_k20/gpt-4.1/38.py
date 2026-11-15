```python
from seed_emulator import (
    Emulator,
    createAutonomousSystem,
    createInternetExchange,
    makeStubAsWithHosts,
    makeStubAs,
    makeEmulatorBaseWith10StubASAndHosts,
    getVirtualNodes,
    createEmulation,
    build_emulator,
    gen_emulation_files,
)
from seed_emulator.layers import EbgpLayer, WebServiceLayer
from seed_emulator.nodes import RouterNode, HostNode, InternalNetwork, WebServiceNode
from seed_emulator.utils import createEmulatorAccountFromMnemonic

# Create the emulator
emulator = Emulator(display_name="AS Emulation Environment")

# --- Transit AS ---
transit_as = createAutonomousSystem(as_type="transit", display_name="Transit AS")

# Create two internet exchanges with specific display names
ix1 = createInternetExchange(display_name="IXP Alpha")
ix2 = createInternetExchange(display_name="IXP Beta")
transit_as.add_internet_exchange(ix1)
transit_as.add_internet_exchange(ix2)

# Create three internal networks
int_net1 = InternalNetwork(display_name="TransitNet1")
int_net2 = InternalNetwork(display_name="TransitNet2")
int_net3 = InternalNetwork(display_name="TransitNet3")
transit_as.add_internal_network(int_net1)
transit_as.add_internal_network(int_net2)
transit_as.add_internal_network(int_net3)

# Create four routers linked in a linear structure
router1 = RouterNode(display_name="TransitRouter1")
router2 = RouterNode(display_name="TransitRouter2")
router3 = RouterNode(display_name="TransitRouter3")
router4 = RouterNode(display_name="TransitRouter4")
transit_as.add_router(router1)
transit_as.add_router(router2)
transit_as.add_router(router3)
transit_as.add_router(router4)

# Link routers linearly
router1.link(router2)
router2.link(router3)
router3.link(router4)

# --- Stub ASes ---
stub_ases = []

# First stub AS: internal network, router, two hosts; first host with extra software and account
stub1 = createAutonomousSystem(as_type="stub", display_name="Stub AS 1")
stub1_net = InternalNetwork(display_name="Stub1Net")
stub1_router = RouterNode(display_name="Stub1Router")
stub1_host1 = HostNode(display_name="Stub1Host1")
stub1_host2 = HostNode(display_name="Stub1Host2")
stub1.add_internal_network(stub1_net)
stub1.add_router(stub1_router)
stub1.add_host(stub1_host1)
stub1.add_host(stub1_host2)
stub1_router.link(stub1_net)
stub1_host1.link(stub1_net)
stub1_host2.link(stub1_net)
# Install extra software and create account on host1
stub1_host1.install_software("nginx")
stub1_host1.create_account(username="webadmin", password="securepass123")
stub_ases.append(stub1)

# Second stub AS: internal network, router, two hosts
stub2 = createAutonomousSystem(as_type="stub", display_name="Stub AS 2")
stub2_net = InternalNetwork(display_name="Stub2Net")
stub2_router = RouterNode(display_name="Stub2Router")
stub2_host1 = HostNode(display_name="Stub2Host1")
stub2_host2 = HostNode(display_name="Stub2Host2")
stub2.add_internal_network(stub2_net)
stub2.add_router(stub2_router)
stub2.add_host(stub2_host1)
stub2.add_host(stub2_host2)
stub2_router.link(stub2_net)
stub2_host1.link(stub2_net)
stub2_host2.link(stub2_net)
stub_ases.append(stub2)

# Third stub AS: use utility function and customize
stub3 = makeStubAsWithHosts(display_name="Stub AS 3", num_hosts=2)
stub3.display_name = "Stub AS 3 (Customized)"
stub3.get_hosts()[0].install_software("curl")
stub_ases.append(stub3)

# --- Utility AS ---
utility_as = createAutonomousSystem(as_type="utility", display_name="Utility AS")

# --- Add ASes to emulator ---
emulator.add_autonomous_system(transit_as)
for stub_as in stub_ases:
    emulator.add_autonomous_system(stub_as)
emulator.add_autonomous_system(utility_as)

# --- BGP Peering ---
ebgp_layer = EbgpLayer(display_name="eBGP Peering")
# Set transit AS as ISP for all stub ASes
for stub_as in stub_ases:
    ebgp_layer.add_isp_customer(transit_as, stub_as)
# Direct peering between first two stub ASes
ebgp_layer.add_peer(stub_ases[0], stub_ases[1])
emulator.add_layer(ebgp_layer)

# --- Web Service Layer ---
web_service_layer = WebServiceLayer(display_name="Web Services")
web_service_node1 = WebServiceNode(display_name="WebService1")
web_service_node2 = WebServiceNode(display_name="WebService2")
web_service_layer.add_web_service(web_service_node1)
web_service_layer.add_web_service(web_service_node2)

# Bind web service nodes to physical hosts
web_service_node1.bind_to(stub1_host1)
web_service_node2.bind_to(stub2_host1)
emulator.add_layer(web_service_layer)

# --- Save to component file ---
emulator.save_to_component_file("as_emulation_env.component")

# --- Render emulator and change display names for web service hosts ---
emulator.render()
stub1_host1.display_name = "Web Host Alpha"
stub2_host1.display_name = "Web Host Beta"

# --- Compile emulator using Docker ---
docker_images = {
    "router": "mydockerhub/router-custom:latest",
    "host": "./local_images/host_image.tar",
    "webservice": "mydockerhub/webservice:2.1"
}
build_emulator(
    emulator=emulator,
    output_folder="./output",
    docker_images=docker_images,
    generate_dockerfiles=True,
    copy_base_image=True
)
```