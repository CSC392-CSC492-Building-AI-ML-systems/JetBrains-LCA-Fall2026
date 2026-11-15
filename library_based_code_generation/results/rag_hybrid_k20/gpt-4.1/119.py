```python
from seed_emulator import (
    createEmulation,
    createAutonomousSystem,
    createHostsOnNetwork,
    _createService,
    TorServer,
    make_dns_as,
    _createServer,
    setRootServers,
    gen_emulation_files,
)
import os
import shutil

# === 1. Create Autonomous Systems and Hosts ===

# Create an AS for the ransomware attacker
attacker_as = createAutonomousSystem("AS100", num_hosts=1)
attacker_host = attacker_as.hosts[0]

# Create an AS for the ransomware victims
victim_as = createAutonomousSystem("AS200", num_hosts=16)
victim_hosts = victim_as.hosts

# Create an AS for Tor nodes
tor_as = createAutonomousSystem("AS300", num_hosts=10)
tor_hosts = tor_as.hosts

# === 2. Ransomware Service ===

# Create ransomware attacker service (no botnet, no Tor)
ransomware_attacker_service = _createService(
    "RansomwareAttacker",
    host=attacker_host,
    config={
        "enable_botnet": False,
        "enable_tor": False,
    }
)

# Create ransomware victim services (no botnet)
ransomware_victim_services = []
for i, host in enumerate(victim_hosts):
    svc = _createService(
        "RansomwareVictim",
        host=host,
        config={
            "enable_botnet": False,
            "victim_id": f"victim_{i+1}"
        }
    )
    ransomware_victim_services.append(svc)

# === 3. Tor Service ===

# Directory Authorities (2)
tor_dir_auth1 = _createService("TorDirectoryAuthority", host=tor_hosts[0])
tor_dir_auth2 = _createService("TorDirectoryAuthority", host=tor_hosts[1])

# Relays (3)
tor_relay1 = _createService("TorRelay", host=tor_hosts[2])
tor_relay2 = _createService("TorRelay", host=tor_hosts[3])
tor_relay3 = _createService("TorRelay", host=tor_hosts[4])

# Exits (2)
tor_exit1 = _createService("TorExit", host=tor_hosts[5])
tor_exit2 = _createService("TorExit", host=tor_hosts[6])

# Clients (2)
tor_client1 = _createService("TorClient", host=tor_hosts[7])
tor_client2 = _createService("TorClient", host=tor_hosts[8])

# Hidden Service (linked to ransomware attacker)
tor_hidden_service = _createService(
    "TorHiddenService",
    host=attacker_host,
    config={
        "service_name": "ransomware_hidden",
        "linked_service": ransomware_attacker_service.name
    }
)

# === 4. DNS Layer ===

# Root DNS Server
root_dns_as = make_dns_as("AS400", num_hosts=1)
root_dns_host = root_dns_as.hosts[0]
root_dns_server = _createServer(
    "DNSRootServer",
    host=root_dns_host,
    config={
        "zones": ["."]  # root zone
    }
)

# TLD DNS Server (.com)
tld_dns_as = make_dns_as("AS401", num_hosts=1)
tld_dns_host = tld_dns_as.hosts[0]
tld_dns_server = _createServer(
    "DNSTLDServer",
    host=tld_dns_host,
    config={
        "zones": [".com"]
    }
)

# ccTLD DNS Server (.us)
cctld_dns_as = make_dns_as("AS402", num_hosts=1)
cctld_dns_host = cctld_dns_as.hosts[0]
cctld_dns_server = _createServer(
    "DNSccTLDServer",
    host=cctld_dns_host,
    config={
        "zones": [".us"]
    }
)

# Second-level zone server (example.com)
sld_dns_as = make_dns_as("AS403", num_hosts=1)
sld_dns_host = sld_dns_as.hosts[0]
sld_dns_server = _createServer(
    "DNSZoneServer",
    host=sld_dns_host,
    config={
        "zones": ["example.com"],
        "records": {
            "www.example.com.": {"A": "192.0.2.10"},
            "attacker.example.com.": {"A": attacker_host.ipv4}
        }
    }
)

# Local DNS server (for victims)
local_dns_as = make_dns_as("AS404", num_hosts=1)
local_dns_host = local_dns_as.hosts[0]
local_dns_server = _createServer(
    "DNSLocalServer",
    host=local_dns_host,
    config={
        "zones": ["local"],
        "records": {
            "victim.local.": {"A": victim_hosts[0].ipv4}
        }
    }
)

# Set root servers for the DNS hierarchy
setRootServers([root_dns_server])

# === 5. Create Emulation ===

emulation = createEmulation(
    name="ransomware-tor-dns-emulation",
    autonomous_systems=[
        attacker_as,
        victim_as,
        tor_as,
        root_dns_as,
        tld_dns_as,
        cctld_dns_as,
        sld_dns_as,
        local_dns_as
    ],
    services=[
        ransomware_attacker_service,
        *ransomware_victim_services,
        tor_dir_auth1, tor_dir_auth2,
        tor_relay1, tor_relay2, tor_relay3,
        tor_exit1, tor_exit2,
        tor_client1, tor_client2,
        tor_hidden_service,
        root_dns_server,
        tld_dns_server,
        cctld_dns_server,
        sld_dns_server,
        local_dns_server
    ]
)

# === 6. Compile Emulator with Docker Compiler and Custom Base Images ===

output_dir = "./emulation_output"
victim_base_image = "./docker/victim_base.Dockerfile"
attacker_base_image = "./docker/attacker_base.Dockerfile"

gen_emulation_files(
    emulation,
    output_dir=output_dir,
    docker_compiler=True,
    custom_base_images={
        "RansomwareVictim": victim_base_image,
        "RansomwareAttacker": attacker_base_image
    }
)

# Copy necessary files to output directory
shutil.copy("start_emulation.sh", os.path.join(output_dir, "start_emulation.sh"))

# Make the script executable
os.chmod(os.path.join(output_dir, "start_emulation.sh"), 0o755)
```