```python
import random
import asyncio
from sc2 import maps, Race, Difficulty, run_game
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2

class ZergRushBot(BotAI):
    async def on_start(self):
        await self.client.game_step(2)
        self.rushed = False
        self.chat_sent = False
        self.inject_targets = set()
        self.creep_debug = True

    async def on_step(self, iteration):
        if not self.chat_sent:
            await self.chat_send("GL HF! Prepare for the swarm!")
            self.chat_sent = True

        await self.manage_resources()
        await self.inject_hatcheries()
        await self.research_upgrades()
        await self.train_units()
        await self.build_structures()
        await self.attack_enemy()
        if self.creep_debug:
            await self.draw_creep_pixelmap()

    async def on_end(self, result):
        print(f"Game ended with result: {result}")

    async def manage_resources(self):
        # Assign drones to gas if we have extractors
        for extractor in self.gas_buildings.ready:
            if extractor.assigned_harvesters < extractor.ideal_harvesters:
                drones = self.workers.closer_than(10, extractor)
                if drones:
                    drone = drones.random
                    await self.do(drone.gather(extractor))
        # Build more drones if we have less than 16 per base and less than 40 total
        if self.supply_workers < 40 and self.can_afford(UnitTypeId.DRONE):
            for hatch in self.townhalls.ready.idle:
                if self.workers.amount < self.townhalls.amount * 16:
                    await self.do(hatch.train(UnitTypeId.DRONE))

    async def inject_hatcheries(self):
        for queen in self.units(UnitTypeId.QUEEN).idle:
            abilities = await self.get_available_abilities(queen)
            if AbilityId.EFFECT_INJECTLARVA in abilities:
                hatch = self.townhalls.ready.closest_to(queen)
                await self.do(queen(AbilityId.EFFECT_INJECTLARVA, hatch))

    async def research_upgrades(self):
        # Research Zergling speed if spawning pool is ready
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            pool = self.structures(UnitTypeId.SPAWNINGPOOL).ready.first
            if not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED):
                if self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED):
                    await self.do(pool.research(UpgradeId.ZERGLINGMOVEMENTSPEED))

    async def train_units(self):
        # Train zerglings from all larva
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            for larva in self.larva:
                if self.can_afford(UnitTypeId.ZERGLING) and self.supply_left > 0:
                    await self.do(larva.train(UnitTypeId.ZERGLING))
        # Train queens for injects
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            for hatch in self.townhalls.ready:
                if self.units(UnitTypeId.QUEEN).amount < self.townhalls.amount and self.can_afford(UnitTypeId.QUEEN):
                    await self.do(hatch.train(UnitTypeId.QUEEN))

    async def build_structures(self):
        # Build spawning pool if we don't have one
        if not self.structures(UnitTypeId.SPAWNINGPOOL) and self.can_afford(UnitTypeId.SPAWNINGPOOL):
            await self.build(UnitTypeId.SPAWNINGPOOL, near=self.townhalls.first.position.towards(self.game_info.map_center, 5))
        # Build extractor if we have a spawning pool and less than 2 extractors
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.gas_buildings.amount < 2:
            for hatch in self.townhalls.ready:
                vgs = self.vespene_geyser.closer_than(10, hatch)
                for vg in vgs:
                    if not self.gas_buildings.closer_than(1, vg):
                        drone = self.workers.closest_to(vg)
                        if self.can_afford(UnitTypeId.EXTRACTOR):
                            await self.do(drone.build(UnitTypeId.EXTRACTOR, vg))
                            return
        # Build overlords if supply is low
        if self.supply_left < 4 and self.can_afford(UnitTypeId.OVERLORD):
            for larva in self.larva:
                await self.do(larva.train(UnitTypeId.OVERLORD))
                break

    async def attack_enemy(self):
        # Attack enemy structures with all zerglings
        if self.units(UnitTypeId.ZERGLING).amount > 10 or self.rushed:
            self.rushed = True
            targets = self.enemy_structures
            if not targets:
                targets = self.enemy_start_locations
            for zergling in self.units(UnitTypeId.ZERGLING).idle:
                if targets:
                    target = targets.closest_to(zergling)
                    await self.do(zergling.attack(target.position if hasattr(target, "position") else target))

    async def draw_creep_pixelmap(self):
        for y in range(0, self.game_info.map_size[1], 4):
            for x in range(0, self.game_info.map_size[0], 4):
                pos = Point2((x, y))
                if self.has_creep(pos):
                    color = (255, 0, 255)
                else:
                    color = (50, 50, 50)
                await self._client.debug_box2_out(pos, pos + Point2((1, 1)), color=color)

if __name__ == "__main__":
    run_game(
        maps.get("AcropolisLE"),
        [
            Bot(Race.Zerg, ZergRushBot()),
            Computer(Race.Terran, Difficulty.Medium)
        ],
        realtime=False,
        save_replay_as="zerg_rush_bot.SC2Replay"
    )
```