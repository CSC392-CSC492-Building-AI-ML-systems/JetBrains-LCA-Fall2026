```python
import random
import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *

class ThreeBaseStargateBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_pylons()
        await self.train_probes()
        await self.expand()
        await self.build_gas()
        await self.build_gateway_and_cybercore()
        await self.build_stargates()
        await self.train_voidrays()
        await self.chrono_boost()
        await self.attack()

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.townhalls.ready
            if nexuses.exists:
                pos = nexuses.random.position.towards(self.game_info.map_center, 5)
                await self.build(PYLON, near=pos)

    async def train_probes(self):
        for nexus in self.townhalls.ready:
            if nexus.is_idle and self.can_afford(PROBE):
                if self.workers.amount < self.townhalls.amount * 22:
                    await self.do(nexus.train(PROBE))

    async def expand(self):
        if self.townhalls.amount < 3 and self.can_afford(NEXUS) and not self.already_pending(NEXUS):
            await self.expand_now()

    async def build_gas(self):
        for nexus in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(15.0, nexus)
            for vg in vgs:
                if not self.gas_buildings.closer_than(1.0, vg).exists and self.can_afford(ASSIMILATOR):
                    worker = self.select_build_worker(vg.position)
                    if worker:
                        await self.do(worker.build(ASSIMILATOR, vg))

    async def build_gateway_and_cybercore(self):
        if self.structures(PYLON).ready.exists:
            pylon = self.structures(PYLON).ready.random
            if self.structures(GATEWAY).amount < 1 and self.can_afford(GATEWAY):
                await self.build(GATEWAY, near=pylon)
            elif self.structures(GATEWAY).ready.exists and self.structures(CYBERNETICSCORE).amount < 1 and self.can_afford(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)

    async def build_stargates(self):
        if self.structures(CYBERNETICSCORE).ready.exists and self.townhalls.ready.amount >= 3:
            if self.structures(STARGATE).amount + self.already_pending(STARGATE) < 3 and self.can_afford(STARGATE):
                pylon = self.structures(PYLON).ready.random
                await self.build(STARGATE, near=pylon)

    async def train_voidrays(self):
        if self.structures(STARGATE).ready.exists and self.townhalls.ready.amount >= 3:
            for stargate in self.structures(STARGATE).ready.idle:
                if self.can_afford(VOIDRAY):
                    await self.do(stargate.train(VOIDRAY))

    async def chrono_boost(self):
        for nexus in self.townhalls.ready:
            abilities = await self.get_available_abilities(nexus)
            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                # Prioritize stargates, then cybercore, then nexus itself
                target = None
                stargates = self.structures(STARGATE).ready
                if stargates.exists:
                    target = stargates.random
                elif self.structures(CYBERNETICSCORE).ready.exists:
                    target = self.structures(CYBERNETICSCORE).ready.random
                else:
                    target = nexus
                await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, target))

    async def attack(self):
        voidrays = self.units(VOIDRAY).ready
        if voidrays.amount > 5:
            for vr in voidrays.idle:
                await self.do(vr.attack(self.enemy_start_locations[0]))
        # If no nexuses left, worker rush
        if not self.townhalls.exists:
            for worker in self.workers.idle:
                await self.do(worker.attack(self.enemy_start_locations[0]))

def main():
    sc2.run_game(
        sc2.maps.get("(2)CatalystLE"),
        [
            Bot(Race.Protoss, ThreeBaseStargateBot()),
            Computer(Race.Protoss, Difficulty.Easy)
        ],
        realtime=False,
    )

if __name__ == "__main__":
    main()
```