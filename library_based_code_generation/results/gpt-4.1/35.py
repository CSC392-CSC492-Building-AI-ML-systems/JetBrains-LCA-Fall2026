```python
import random
import asyncio
from sc2 import maps, Race, Difficulty, run_game, position
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

class StalkerBot(BotAI):
    async def on_step(self, iteration):
        if not self.townhalls:
            # No Nexuses left, try to fight with remaining units
            for stalker in self.units(UnitTypeId.STALKER).idle:
                targets = self.enemy_units | self.enemy_structures
                if targets:
                    stalker.attack(targets.closest_to(stalker))
                else:
                    stalker.attack(self.enemy_start_locations[0])
            return

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

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            nex = self.townhalls.ready.random
            pos = nex.position.towards(self.game_info.map_center, 5)
            if self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=pos)

    async def build_assimilators(self):
        for nexus in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(15.0, nexus)
            for vg in vgs:
                if not self.gas_buildings.closer_than(1.0, vg).exists and self.can_afford(UnitTypeId.ASSIMILATOR):
                    worker = self.select_build_worker(vg.position)
                    if worker:
                        await self.do(worker.build(UnitTypeId.ASSIMILATOR, vg))

    async def train_probes(self):
        for nexus in self.townhalls.ready.idle:
            if self.can_afford(UnitTypeId.PROBE) and self.workers.amount < self.townhalls.amount * 22:
                await self.do(nexus.train(UnitTypeId.PROBE))

    async def build_gateway(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            if self.structures(UnitTypeId.GATEWAY).amount + self.structures(UnitTypeId.WARPGATE).amount < 2 and self.can_afford(UnitTypeId.GATEWAY):
                pylon = self.structures(UnitTypeId.PYLON).ready.random
                await self.build(UnitTypeId.GATEWAY, near=pylon)

    async def build_cybercore(self):
        if self.structures(UnitTypeId.GATEWAY).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE):
            if self.can_afford(UnitTypeId.CYBERNETICSCORE):
                pylon = self.structures(UnitTypeId.PYLON).ready.random
                await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)

    async def research_warpgate(self):
        for core in self.structures(UnitTypeId.CYBERNETICSCORE).ready.idle:
            abilities = await self.get_available_abilities(core)
            if AbilityId.RESEARCH_WARPGATE in abilities and self.can_afford(RESEARCH_WARPGATE):
                await self.do(core(AbilityId.RESEARCH_WARPGATE))

    async def morph_gateways(self):
        if self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) or self.structures(UnitTypeId.WARPGATE).exists:
            for gw in self.structures(UnitTypeId.GATEWAY).ready:
                abilities = await self.get_available_abilities(gw)
                if AbilityId.MORPH_WARPGATE in abilities:
                    await self.do(gw(AbilityId.MORPH_WARPGATE))

    async def build_proxy_pylon(self):
        # Build proxy pylon near enemy base if not already built
        if self.structures(UnitTypeId.PYLON).amount < 3 and self.can_afford(UnitTypeId.PYLON):
            proxy_location = self.enemy_start_locations[0].towards(self.game_info.map_center, 20)
            if not self.structures(UnitTypeId.PYLON).closer_than(10, proxy_location).exists:
                await self.build(UnitTypeId.PYLON, near=proxy_location)

    async def warp_stalkers(self):
        if self.structures(UnitTypeId.WARPGATE).ready.exists:
            for warpgate in self.structures(UnitTypeId.WARPGATE).ready:
                abilities = await self.get_available_abilities(warpgate)
                if AbilityId.WARPGATETRAIN_STALKER in abilities and self.can_afford(UnitTypeId.STALKER):
                    pylon = self.structures(UnitTypeId.PYLON).ready.closest_to(self.enemy_start_locations[0])
                    placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pylon.position, placement_step=2)
                    if placement:
                        await self.do(warpgate.warp_in(UnitTypeId.STALKER, placement))

    async def chrono_boost(self):
        for nexus in self.townhalls.ready:
            abilities = await self.get_available_abilities(nexus)
            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                # Prioritize Cybercore, then Nexus
                target = None
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                    core = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
                    if not core.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        target = core
                if not target:
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        target = nexus
                if target:
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, target))

    async def stalker_attack(self):
        stalkers = self.units(UnitTypeId.STALKER).idle
        if stalkers.exists:
            targets = self.enemy_units | self.enemy_structures
            if targets.exists:
                for stalker in stalkers:
                    await self.do(stalker.attack(targets.closest_to(stalker)))
            else:
                for stalker in stalkers:
                    await self.do(stalker.attack(self.enemy_start_locations[0]))

def main():
    run_game(
        maps.get("(2)CatalystLE"),
        [Bot(Race.Protoss, StalkerBot()), Computer(Race.Protoss, Difficulty.Easy)],
        realtime=False,
    )

if __name__ == "__main__":
    main()
```