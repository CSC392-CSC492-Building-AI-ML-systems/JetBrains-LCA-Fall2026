```python
import random
import asyncio
from sc2 import maps, run_game, Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2

class ZergRushBot(BotAI):
    async def on_start(self):
        await self.client.chat_send("GL HF! Zerg Rush incoming!", team_only=False)
        await self.client.game_step(2)
        self.rush_sent = False

    async def on_step(self, iteration):
        # Send a chat message at the start
        if iteration == 2:
            await self.client.chat_send("You must construct additional pylons! (Just kidding, Zerg rush!)", team_only=False)

        await self.inject_larva()
        await self.manage_resources()
        await self.research_upgrades()
        await self.train_units()
        await self.build_structures()
        await self.attack_enemy()
        await self.draw_creep_pixelmap()

    async def on_end(self, result):
        print(f"Game ended with result: {result}")

    async def inject_larva(self):
        for queen in self.units(UnitTypeId.QUEEN).idle:
            abilities = await self.get_available_abilities(queen)
            for hatch in self.townhalls.ready:
                if AbilityId.EFFECT_INJECTLARVA in abilities and queen.energy >= 25:
                    if queen.distance_to(hatch) < 8:
                        queen(AbilityId.EFFECT_INJECTLARVA, hatch)
                        break

    async def manage_resources(self):
        # Build Extractors if we have at least 2 hatcheries and less than 2 extractors per base
        for hatch in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(10, hatch)
            for vg in vgs:
                if self.can_afford(UnitTypeId.EXTRACTOR):
                    worker = self.select_build_worker(vg.position)
                    if worker and not self.gas_buildings.closer_than(1, vg):
                        worker.build(UnitTypeId.EXTRACTOR, vg)
                        break

        # Saturate minerals and gas
        for hatch in self.townhalls.ready:
            if self.workers.amount < self.townhalls.amount * 22 and self.can_afford(UnitTypeId.DRONE):
                self.train(UnitTypeId.DRONE)

    async def research_upgrades(self):
        # Research Zergling Speed
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED):
            pool = self.structures(UnitTypeId.SPAWNINGPOOL).ready.first
            if not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED):
                if pool.is_idle:
                    pool.research(UpgradeId.ZERGLINGMOVEMENTSPEED)

    async def train_units(self):
        # Train Zerglings
        for hatch in self.townhalls.ready.idle:
            if self.can_afford(UnitTypeId.ZERGLING) and self.supply_left > 0:
                hatch.train(UnitTypeId.ZERGLING)
        # Train Queens
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.units(UnitTypeId.QUEEN).amount < self.townhalls.amount:
            for hatch in self.townhalls.ready:
                if self.can_afford(UnitTypeId.QUEEN) and hatch.is_idle:
                    hatch.train(UnitTypeId.QUEEN)

    async def build_structures(self):
        # Build Spawning Pool if not exists
        if not self.structures(UnitTypeId.SPAWNINGPOOL) and self.can_afford(UnitTypeId.SPAWNINGPOOL):
            location = await self.get_next_expansion()
            drone = self.select_build_worker(location)
            if drone:
                pos = await self.find_placement(UnitTypeId.SPAWNINGPOOL, near=location, placement_step=3)
                if pos:
                    drone.build(UnitTypeId.SPAWNINGPOOL, pos)
        # Build Overlords if supply is low
        if self.supply_left < 4 and self.can_afford(UnitTypeId.OVERLORD):
            for hatch in self.townhalls.ready.idle:
                hatch.train(UnitTypeId.OVERLORD)

    async def attack_enemy(self):
        # Send all Zerglings to attack enemy structures as soon as we have at least 16
        if self.units(UnitTypeId.ZERGLING).amount >= 16 or self.rush_sent:
            self.rush_sent = True
            targets = self.enemy_structures
            if targets:
                target = targets.closest_to(self.start_location)
            else:
                target = self.enemy_start_locations[0]
            for zergling in self.units(UnitTypeId.ZERGLING).idle:
                zergling.attack(target)

    async def draw_creep_pixelmap(self):
        # Draw creep as purple pixels for debugging
        for y in range(self.game_info.map_size[1]):
            for x in range(self.game_info.map_size[0]):
                pos = Point2((x, y))
                if self.state.creep_cover_map.bool_map[y][x]:
                    self._client.debug_text_world("•", pos, color=(128, 0, 128), size=8)

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