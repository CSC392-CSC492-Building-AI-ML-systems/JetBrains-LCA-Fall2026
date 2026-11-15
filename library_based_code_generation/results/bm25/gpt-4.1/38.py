```python
from seed_emulator import (
    Emulator,
    makeTransitAs,
    makeStubAsWithHosts,
    makeStubAs,
    make_service_as,
    Ebgp,
    WebService,
    getNodeByAsnAndName,
    getVirtualNodes,
    registerNodes,
    up_emulator,
    getNodesByName,
)

# Create Emulator
emu = Emulator()

# --- Transit AS ---
transit_as = makeTransitAs(
    asn=65000,
    num_routers=4,
    num_internal_networks=3,
    num_ix=2,
    ix_display_names=["IX-West", "IX-East"],
    router_link_structure="linear"
)
transit_as.display_name = "Transit-AS"
emu.add_layer(transit_as)

# --- Stub ASes ---
# Stub AS 1
stub1 = makeStubAsWithHosts(
    asn=65001,
    num_hosts=2,
    num_internal_networks=1,
    router_name="r1"
)
stub1.display_name = "Stub-AS-1"
host1_1 = getNodeByAsnAndName(65001, "host1")
host1_1.install_software("nginx")
host1_1.create_account("webadmin", password="securepass123")

# Stub AS 2
stub2 = makeStubAsWithHosts(
    asn=65002,
    num_hosts=2,
    num_internal_networks=1,
    router_name="r2"
)
stub2.display_name = "Stub-AS-2"

# Stub AS 3 (using utility function and further customized)
stub3 = makeStubAs(asn=65003)
stub3.add_internal_network("net1")
stub3.add_router("r3")
stub3.add_host("host3-1")
stub3.add_host("host3-2")
stub3.display_name = "Stub-AS-3"
# Customization: install software on host3-1
host3_1 = getNodeByAsnAndName(65003, "host3-1")
host3_1.install_software("curl")

emu.add_layer(stub1)
emu.add_layer(stub2)
emu.add_layer(stub3)

# --- BGP Peering ---
ebgp = Ebgp()
# Transit AS as ISP for all stub ASes
ebgp.add_peering(provider_asn=65000, customer_asn=65001)
ebgp.add_peering(provider_asn=65000, customer_asn=65002)
ebgp.add_peering(provider_asn=65000, customer_asn=65003)
# Direct peering between stub1 and stub2
ebgp.add_peering(peer1_asn=65001, peer2_asn=65002)
emu.add_layer(ebgp)

# --- Web Service Layer ---
web_service_layer = WebService()
web_service_node1 = web_service_layer.add_service_node("websvc1")
web_service_node2 = web_service_layer.add_service_node("websvc2")
# Bind virtual nodes to physical nodes
physical_node1 = getNodeByAsnAndName(65001, "host1")
physical_node2 = getNodeByAsnAndName(65002, "host2")
web_service_layer.bind_virtual_to_physical("websvc1", physical_node1)
web_service_layer.bind_virtual_to_physical("websvc2", physical_node2)
emu.add_layer(web_service_layer)

# --- Save to component file ---
emu.save_to_component_file("my_emulator_component.yaml")

# --- Render Emulator ---
emu.render()

# --- Change display names for nodes hosting web services ---
websvc_nodes = getNodesByName(["websvc1", "websvc2"])
websvc_nodes[0].display_name = "Web Service Node 1"
websvc_nodes[1].display_name = "Web Service Node 2"

# --- Compile Emulator using Docker ---
emu.compile_with_docker(
    docker_images={
        "nginx": "nginx:1.25-alpine",
        "curl": "curlimages/curl:latest",
        "base": "./local_base_image"
    },
    output_folder="./output"
)
emu.generate_docker_files(output_folder="./output")
emu.copy_base_container_image("./local_base_image", "./output")
```