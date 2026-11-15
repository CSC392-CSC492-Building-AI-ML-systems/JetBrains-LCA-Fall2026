```python
import asyncio
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty, Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.main import run_game

class ZergRushBot(BotAI):
    async def on_start(self):
        await self.client.chat_send("GL HF! Prepare for the Zerg rush!", team_only=False)
        await self.client.debug_game_step(2)
        self.rushed = False

    async def on_step(self, iteration):
        # Send a chat message at the start of the rush
        if not self.rushed and self.supply_used >= 14:
            await self.client.chat_send("Zerg rush incoming!", team_only=False)
            self.rushed = True

        # Inject hatcheries with larva
        for queen in self.units(UnitTypeId.QUEEN).idle:
            hatch = self.townhalls.ready.random_or(None)
            if hatch and queen.energy >= 25:
                abilities = await self.get_available_abilities(queen)
                if AbilityId.EFFECT_INJECTLARVA in abilities:
                    queen(AbilityId.EFFECT_INJECTLARVA, hatch)

        # Build overlords if supply is low
        if self.supply_left < 2 and self.can_afford(UnitTypeId.OVERLORD):
            if self.units(UnitTypeId.LARVA).exists:
                self.units(UnitTypeId.LARVA).random.train(UnitTypeId.OVERLORD)

        # Build spawning pool if not exists
        if not self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.can_afford(UnitTypeId.SPAWNINGPOOL):
            pos = await self.get_next_expansion()
            await self.build(UnitTypeId.SPAWNINGPOOL, near=pos)

        # Build extractor if spawning pool exists and no extractor
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists and self.vespene < 100:
            for hatch in self.townhalls.ready:
                vgs = self.vespene_geyser.closer_than(10, hatch)
                for vg in vgs:
                    if not self.gas_buildings.closer_than(1, vg).exists and self.can_afford(UnitTypeId.EXTRACTOR):
                        drone = self.workers.closer_than(10, vg).random_or(None)
                        if drone:
                            drone.build(UnitTypeId.EXTRACTOR, vg)
                            break

        # Train zerglings
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            for larva in self.units(UnitTypeId.LARVA):
                if self.can_afford(UnitTypeId.ZERGLING):
                    larva.train(UnitTypeId.ZERGLING)

        # Research metabolic boost (zergling speed)
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            pool = self.structures(UnitTypeId.SPAWNINGPOOL).ready.first
            if self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED) == 0:
                abilities = await self.get_available_abilities(pool)
                if AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST in abilities and self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED):
                    pool(AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST)

        # Build more hatcheries if minerals allow
        if self.townhalls.amount < 2 and self.can_afford(UnitTypeId.HATCHERY):
            pos = await self.get_next_expansion()
            await self.build(UnitTypeId.HATCHERY, near=pos)

        # Attack enemy structures with zerglings
        if self.units(UnitTypeId.ZERGLING).amount > 10:
            targets = self.enemy_structures
            if targets.exists:
                for zergling in self.units(UnitTypeId.ZERGLING).idle:
                    zergling.attack(targets.closest_to(zergling))
            else:
                for zergling in self.units(UnitTypeId.ZERGLING).idle:
                    zergling.attack(self.enemy_start_locations[0])

        # Draw creep pixelmap for debugging
        await self.draw_creep_pixelmap()

    async def draw_creep_pixelmap(self):
        for y in range(0, self.game_info.map_size[1], 8):
            for x in range(0, self.game_info.map_size[0], 8):
                pos = Point2((x, y))
                if self.has_creep(pos):
                    color = (255, 0, 255)
                    await self.client.debug_box2_out(pos, pos + Point2((8, 8)), color=color)

    async def on_end(self, result: Result):
        print(f"Game ended with result: {result}")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(
        run_game(
            maps.get("AcropolisLE"),
            [
                Bot(Race.Zerg, ZergRushBot()),
                Computer(Race.Terran, Difficulty.Medium)
            ],
            realtime=False,
            save_replay_as="zerg_rush_bot.SC2Replay"
        )
    )
```