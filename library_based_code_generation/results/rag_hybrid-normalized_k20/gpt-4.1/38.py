```python
from seed_emulator import (
    Emulator,
    createAutonomousSystem,
    createInternetExchange,
    makeStubAsWithHosts,
    makeStubAs,
    createEmulation,
    getVirtualNodes,
    build_emulator,
    gen_emulation_files,
)
from seed_emulator.layers import EbgpLayer, WebServiceLayer
from seed_emulator.nodes import RouterNode, HostNode, NetworkNode, WebServiceNode
from seed_emulator.utils import createEmulatorAccountFromMnemonic

# Create Emulator
emulator = Emulator(display_name="AS Emulation Environment")

# --- Transit AS ---
transit_as = createAutonomousSystem(as_type="transit", display_name="Transit AS")

# Create two Internet Exchanges
ix1 = createInternetExchange(display_name="IX Alpha")
ix2 = createInternetExchange(display_name="IX Beta")
transit_as.add_internet_exchange(ix1)
transit_as.add_internet_exchange(ix2)

# Create three internal networks
net1 = NetworkNode(display_name="TransitNet1")
net2 = NetworkNode(display_name="TransitNet2")
net3 = NetworkNode(display_name="TransitNet3")
transit_as.add_network(net1)
transit_as.add_network(net2)
transit_as.add_network(net3)

# Create four routers in a linear structure
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

# Stub AS 1
stub1 = createAutonomousSystem(as_type="stub", display_name="Stub AS 1")
stub1_net = NetworkNode(display_name="Stub1Net")
stub1_router = RouterNode(display_name="Stub1Router")
stub1_host1 = HostNode(display_name="Stub1Host1")
stub1_host2 = HostNode(display_name="Stub1Host2")
stub1.add_network(stub1_net)
stub1.add_router(stub1_router)
stub1.add_host(stub1_host1)
stub1.add_host(stub1_host2)
# Install additional software and create new account on host1
stub1_host1.install_software("nginx")
stub1_host1.create_account(username="webadmin", password="securepass")
stub_ases.append(stub1)

# Stub AS 2
stub2 = createAutonomousSystem(as_type="stub", display_name="Stub AS 2")
stub2_net = NetworkNode(display_name="Stub2Net")
stub2_router = RouterNode(display_name="Stub2Router")
stub2_host1 = HostNode(display_name="Stub2Host1")
stub2_host2 = HostNode(display_name="Stub2Host2")
stub2.add_network(stub2_net)
stub2.add_router(stub2_router)
stub2.add_host(stub2_host1)
stub2.add_host(stub2_host2)
stub_ases.append(stub2)

# Stub AS 3 using utility function and customize
stub3 = makeStubAsWithHosts(display_name="Stub AS 3", num_hosts=2)
stub3.display_name = "Stub AS 3 Customized"
stub3.hosts[0].install_software("apache2")
stub_ases.append(stub3)

# --- Utility AS ---
utility_as = createAutonomousSystem(as_type="utility", display_name="Utility AS")

# --- BGP Peering ---
ebgp_layer = EbgpLayer(display_name="Ebgp Peering")
# Set transit AS as ISP for all stub ASes
for stub_as in stub_ases:
    ebgp_layer.add_peering(transit_as, stub_as)
# Direct peering between stub1 and stub2
ebgp_layer.add_peering(stub1, stub2)

# --- Web Service Layer ---
web_service_layer = WebServiceLayer(display_name="Web Services")
web_service_node1 = WebServiceNode(display_name="WebServiceNode1")
web_service_node2 = WebServiceNode(display_name="WebServiceNode2")
web_service_layer.add_service(web_service_node1)
web_service_layer.add_service(web_service_node2)

# Bind web service nodes to physical hosts
web_service_node1.bind_to(stub1_host1)
web_service_node2.bind_to(stub2_host1)

# --- Add all layers to emulator ---
emulator.add_autonomous_system(transit_as)
emulator.add_autonomous_system(stub1)
emulator.add_autonomous_system(stub2)
emulator.add_autonomous_system(stub3)
emulator.add_autonomous_system(utility_as)
emulator.add_layer(ebgp_layer)
emulator.add_layer(web_service_layer)

# --- Save to component file ---
emulator.save_to_file("as_emulation_component.json")

# --- Render emulator and change display names for web service hosts ---
emulator.render()
stub1_host1.display_name = "Web Host Alpha"
stub2_host1.display_name = "Web Host Beta"

# --- Compile emulator using Docker ---
docker_images = {
    "router": "mydockerhub/router-custom:latest",
    "host": "./local_images/host-custom.Dockerfile",
    "webservice": "mydockerhub/webservice:2.1"
}
build_emulator(
    emulator,
    docker_images=docker_images,
    output_folder="./emulator_output",
    generate_dockerfiles=True,
    copy_base_image=True
)
```