```python
import random
import asyncio
from sc2 import maps, run_game, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.bot_ai import BotAI
from sc2.position import Point2

PROXY_PYLON_DISTANCE = 30

class StalkerBot(BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.train_probes()
        await self.build_gateway()
        await self.build_cybercore()
        await self.research_warpgate()
        await self.morph_gateways()
        await self.build_proxy_pylon()
        await self.warp_stalkers()
        await self.chrono_boost()
        await self.stalker_attack()
        await self.handle_no_nexus()

    async def build_pylons(self):
        if self.supply_left < 5 and self.already_pending(PYLON) == 0:
            nexuses = self.townhalls.ready
            if nexuses.exists:
                await self.build(PYLON, near=nexuses.first.position.towards(self.game_info.map_center, 5))

    async def build_assimilators(self):
        for nexus in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(15.0, nexus)
            for vg in vgs:
                if not self.gas_buildings.closer_than(1.0, vg).exists and not self.already_pending(ASSIMILATOR):
                    worker = self.select_build_worker(vg.position)
                    if worker and self.can_afford(ASSIMILATOR):
                        await self.do(worker.build(ASSIMILATOR, vg))

    async def train_probes(self):
        for nexus in self.townhalls.ready.idle:
            if self.can_afford(PROBE) and self.workers.amount < self.townhalls.amount * 22:
                await self.do(nexus.train(PROBE))

    async def build_gateway(self):
        if self.structures(PYLON).ready.exists:
            if self.structures(GATEWAY).amount + self.already_pending(GATEWAY) < 2:
                pylon = self.structures(PYLON).ready.random
                if self.can_afford(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

    async def build_cybercore(self):
        if self.structures(GATEWAY).ready.exists and not self.structures(CYBERNETICSCORE):
            pylon = self.structures(PYLON).ready.random
            if self.can_afford(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)

    async def research_warpgate(self):
        for core in self.structures(CYBERNETICSCORE).ready.idle:
            if self.can_afford(RESEARCH_WARPGATE) and not self.already_pending_upgrade(WARPGATERESEARCH):
                await self.do(core(RESEARCH_WARPGATE))

    async def morph_gateways(self):
        if self.already_pending_upgrade(WARPGATERESEARCH) == 0 and self.structures(CYBERNETICSCORE).ready.exists:
            if self.structures(CYBERNETICSCORE).ready.first().is_researching:
                return
        for gateway in self.structures(GATEWAY).ready:
            abilities = await self.get_available_abilities(gateway)
            if AbilityId.MORPH_WARPGATE in abilities:
                await self.do(gateway(AbilityId.MORPH_WARPGATE))

    async def build_proxy_pylon(self):
        if self.structures(PYLON).amount < 2 and self.can_afford(PYLON):
            enemy_pos = self.enemy_start_locations[0]
            pos = self.start_location.towards(enemy_pos, PROXY_PYLON_DISTANCE)
            if not self.structures(PYLON).closer_than(8, pos).exists and not self.already_pending(PYLON):
                await self.build(PYLON, near=pos)

    async def warp_stalkers(self):
        if self.structures(WARPGATE).ready.exists and self.structures(PYLON).ready.exists:
            for warpgate in self.structures(WARPGATE).ready:
                abilities = await self.get_available_abilities(warpgate)
                if AbilityId.WARPGATETRAIN_STALKER in abilities and self.can_afford(STALKER):
                    pylon = self.structures(PYLON).ready.closest_to(self.enemy_start_locations[0])
                    placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pylon.position, placement_step=1)
                    if placement:
                        await self.do(warpgate.warp_in(STALKER, placement))

    async def chrono_boost(self):
        for nexus in self.townhalls.ready:
            if nexus.energy >= 50:
                if self.structures(CYBERNETICSCORE).ready.exists and not self.structures(CYBERNETICSCORE).ready.first().is_researching:
                    continue
                if self.structures(CYBERNETICSCORE).ready.exists:
                    core = self.structures(CYBERNETICSCORE).ready.first()
                    if not core.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, core))
                        return
                if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus))
                    return

    async def stalker_attack(self):
        stalkers = self.units(STALKER).idle
        if stalkers.exists:
            targets = self.enemy_units
            if targets.exists:
                for stalker in stalkers:
                    target = targets.closest_to(stalker)
                    await self.do(stalker.attack(target))
            else:
                for stalker in stalkers:
                    await self.do(stalker.attack(self.enemy_start_locations[0]))

    async def handle_no_nexus(self):
        if not self.townhalls.exists:
            for unit in self.units:
                if unit.can_attack:
                    await self.do(unit.attack(self.enemy_start_locations[0]))

def main():
    run_game(
        maps.get("(2)CatalystLE"),
        [Bot(Race.Protoss, StalkerBot()), Computer(Race.Protoss, Difficulty.Easy)],
        realtime=False,
    )

if __name__ == "__main__":
    main()
```