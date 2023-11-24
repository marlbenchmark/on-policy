import numpy as np
from onpolicy.envs.overcooked.script_agent.base import BaseScriptAgent
from onpolicy.envs.overcooked.script_agent.script_period import SCRIPT_PERIODS_CLASSES
import onpolicy.envs.overcooked.script_agent.utils as utils
from onpolicy.envs.overcooked.overcooked_ai_py.mdp.actions import Direction, Action
import random
import functools

class RandomScriptAgent(BaseScriptAgent):
    def __init__(self, periods_config):
        super().__init__()
        self.periods_config = periods_config
        self.period_name = [p for p in periods_config.keys()]
        self.probs = np.array([d["prob"] for p, d in periods_config.items()])
        self.probs /= self.probs.sum()

    def make_new_period(self, i=None):
        if i is None:
            i = np.random.choice(np.arange(len(self.period_name)), p=self.probs)
        elif type(i) == list:
            p = self.probs[i].copy()
            p = p / p.sum()
            i = np.random.choice(i, p=p)
        p = self.period_name[i]
        return p, SCRIPT_PERIODS_CLASSES[p](**self.periods_config[p]["args"])

    def reset(self, mdp, state, player_idx):
        """reset state
        """
        self._current_period_name, self._current_period = self.make_new_period()
        self._current_period.reset(mdp, state, player_idx)
        self.last_pos = state.players[player_idx].position
        self.stuck_time = 0

    def step(self, mdp, state, player_idx):
        # print(f"step {player_idx}\n", mdp.state_string(state))
        while self._current_period.done(mdp, state, player_idx):
            self._current_period_name, self._current_period = self.make_new_period()
            self._current_period.reset(mdp, state, player_idx)
        action = self._current_period.step(mdp, state, player_idx)
        pos = state.players[player_idx].position
        if pos == self.last_pos:
            self.stuck_time += 1
            if self.stuck_time >= 3:
                action = random.choice(Direction.ALL_DIRECTIONS)
        else:
            self.last_pos = pos
            self.stuck_time = 0
        # print(self._current_period_name, action)
        return action

class Place_Onion_in_Pot_Agent(RandomScriptAgent):
    def __init__(self):
        super().__init__(
            {
                "pickup_onion_and_place_in_pot": dict(prob=1., args=dict()),
            }
        )

class Deliver_Soup_Agent(RandomScriptAgent):
    def __init__(self):
        super().__init__(
            {
                "pickup_soup_and_deliver": dict(prob=1., args=dict()),
            }
        )

class SinglePeriodScriptAgent(RandomScriptAgent):
    def __init__(self, period_name):
        super().__init__(
            {
                period_name: dict(prob=1., args=dict()),
            }
        )

class Place_Onion_and_Deliver_Soup_Agent(RandomScriptAgent):
    def __init__(self):
        super().__init__(
            {
                "pickup_onion_and_place_in_pot": dict(prob=0.5, args=dict()),
                "pickup_soup_and_deliver": dict(prob=0.5, args=dict()),
            }
        )
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]
        if self._current_period_name == "pickup_onion_and_place_in_pot":
            if not utils.exists(mdp, state, player_idx, terrain_type="P", obj=["empty", "unfull_soup"]):
                # no available space to put onion
                self._current_period_name, self._current_period = self.make_new_period(i=1) # pickup_soup_and_deliver
                self._current_period.reset(mdp, state, player_idx)
        if self._current_period_name == "pickup_soup_and_deliver":
            if not (player.has_object() and player.get_object().name == "soup") and not utils.exists(mdp, state, player_idx, terrain_type="P", obj=["soup", "cooking_soup"]):
                # no cooking soup or ready soup, should pick onion and place in pot
                self._current_period_name, self._current_period = self.make_new_period(i=0) # pickup_onion_and_place_in_pot
                self._current_period.reset(mdp, state, player_idx)
        return super(Place_Onion_and_Deliver_Soup_Agent, self).step(mdp, state, player_idx)
            
class Noisy_Agent(RandomScriptAgent):
    def __init__(self, onion_ratio, soup_ratio, noise_ratio):
        super().__init__(
            {
                "pickup_onion_and_place_in_pot": dict(prob=onion_ratio * (1. - noise_ratio), args=dict()),
                "pickup_onion_and_place_random": dict(prob=onion_ratio * noise_ratio, args=dict()),
                "pickup_soup_and_deliver": dict(prob=soup_ratio * (1. - noise_ratio), args=dict()),
                "pickup_soup_and_place_random": dict(prob=soup_ratio * noise_ratio, args=dict()),
            }
        )
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]
        if 'pickup_onion' in self._current_period_name:
            if not utils.exists(mdp, state, player_idx, terrain_type="P", obj=["empty", "unfull_soup"]):
                # no available space to put onion, take soup away
                self._current_period_name, self._current_period = self.make_new_period(i=[2, 3]) # pickup_soup_and_deliver or pickup_soup_and_place_random
                self._current_period.reset(mdp, state, player_idx)
        if self._current_period_name == "pickup_soup_and_deliver":
            if not (player.has_object() and player.get_object().name == "soup") and not utils.exists(mdp, state, player_idx, terrain_type="P", obj=["soup", "cooking_soup"]):
                # no cooking soup or ready soup, should pick onion and place in pot
                self._current_period_name, self._current_period = self.make_new_period(i=0) # pickup_onion_and_place_in_pot; Do not use pickup_onion_and_place_random since we want to really put onion into pot
                self._current_period.reset(mdp, state, player_idx)
        return super(Noisy_Agent, self).step(mdp, state, player_idx)


class Random3_Only_Onion_to_Middle_Agent(RandomScriptAgent):
    def __init__(self):
        super().__init__(
            {
                "random3_only_onion_to_middle": dict(prob=1., args=dict())
            }
        )

SCRIPT_AGENTS = {
    "place_onion_in_pot":  functools.partial(SinglePeriodScriptAgent, period_name="pickup_onion_and_place_in_pot"),
    "deliver_soup": functools.partial(SinglePeriodScriptAgent, period_name="pickup_soup_and_deliver"),
    "place_onion_and_deliver_soup": Place_Onion_and_Deliver_Soup_Agent,
    # "noisy": Noisy_Agent,
    "random3_only_onion_to_middle": Random3_Only_Onion_to_Middle_Agent,
    "put_onion_everywhere": functools.partial(SinglePeriodScriptAgent, period_name="put_onion_everywhere"),
    "put_dish_everywhere": functools.partial(SinglePeriodScriptAgent, period_name="put_dish_everywhere")
}

for onion in range(10 + 1):
    for noise in range(10 + 1):
        soup = 10 - onion
        onion_ratio = onion / 10
        noise_ratio = noise / 10
        soup_ratio = 1. - onion_ratio
        SCRIPT_AGENTS[f"{onion}onion_{soup}soup_{noise}noise"] = functools.partial(Noisy_Agent, onion_ratio=onion_ratio, soup_ratio=soup_ratio, noise_ratio=noise_ratio)