from hsp.envs.overcooked.overcooked_ai_py.mdp.actions import Direction, Action
from hsp.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, ObjectState, PlayerState
from hsp.envs.overcooked.overcooked_ai_py.agents.benchmarking import AgentEvaluator

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
S, P, Obj = OvercookedState, PlayerState, ObjectState

delivery_reward = 20

s_a_r_pairs = [
    (S([P((1, 2), n), P((3, 1), n)], {}, order_list=['onion', 'any']), [n, e], 0),
    (S([P((1, 1), n), P((3, 1), e)], {}, order_list=['onion', 'any']), [w, interact], 0),
    (S([P((1, 1), w), P((3, 1), e, Obj('onion', (3, 1)))], {}, order_list=['onion', 'any']), [interact, w], 0),
    (S([P((1, 1), w, Obj('onion', (1, 1))),P((2, 1), w, Obj('onion', (2, 1)))],{}, order_list=['onion', 'any']), [e, n], 0),
    (S([P((1, 1), e, Obj('onion', (1, 1))),P((2, 1), n, Obj('onion', (2, 1)))],{}, order_list=['onion', 'any']), [stay, interact], 0),
    (S([P((1, 1), e, Obj('onion', (1, 1))),P((2, 1), n)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},order_list=['onion', 'any']), [e, e], 0),
    (S([P((2, 1), e, Obj('onion', (2, 1))),P((3, 1), e)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},order_list=['onion', 'any']), [n, interact], 0),
    (S([P((2, 1), n, Obj('onion', (2, 1))),P((3, 1), e, Obj('onion', (3, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},order_list=['onion', 'any']), [interact, w], 0),
    (S([P((2, 1), n),P((3, 1), w, Obj('onion', (3, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 2, 0))},order_list=['onion', 'any']), [w, w], 0),
    (S([P((1, 1), w),P((2, 1), w, Obj('onion', (2, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 2, 0))},order_list=['onion', 'any']), [s, n], 0),
    (S([P((1, 2), s),P((2, 1), n, Obj('onion', (2, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 2, 0))},order_list=['onion', 'any']), [interact, interact], 0),
    (S([P((1, 2), s, Obj('dish', (1, 2))),P((2, 1), n)],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 1))},order_list=['onion', 'any']), [e, s], 0),
    (S([P((1, 2), e, Obj('dish', (1, 2))),P((2, 1), s)],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 2))},order_list=['onion', 'any']), [e, interact], 0),
    (S([P((2, 2), e, Obj('dish', (2, 2))),P((2, 1), s)],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 3))},order_list=['onion', 'any']), [n, e], 0),
    (S([P((2, 1), n, Obj('dish', (2, 1))),P((3, 1), e)],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 4))},order_list=['onion', 'any']), [interact, interact], 0),
    (S([P((2, 1), n, Obj('dish', (2, 1))),P((3, 1), e, Obj('onion', (3, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 5))},order_list=['onion', 'any']), [stay, stay], 0),
    (S([P((2, 1), n, Obj('dish', (2, 1))),P((3, 1), e, Obj('onion', (3, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 5))},order_list=['onion', 'any']), [interact, interact], 0),
    (S([P((2, 1), n, Obj('soup', (2, 1), ('onion', 3, 5))),P((3, 1), e, Obj('onion', (3, 1)))],{}, order_list=['onion', 'any']), [e, w], 0),
    (S([P((2, 1), e, Obj('soup', (2, 1), ('onion', 3, 5))),P((3, 1), w, Obj('onion', (3, 1)))],{}, order_list=['onion', 'any']), [e, s], 0),
    (S([P((3, 1), e, Obj('soup', (3, 1), ('onion', 3, 5))),P((3, 2), s, Obj('onion', (3, 2)))],{}, order_list=['onion', 'any']), [s, interact], 0),
    (S([P((3, 1), s, Obj('soup', (3, 1), ('onion', 3, 5))),P((3, 2), s, Obj('onion', (3, 2)))],{}, order_list=['onion', 'any']), [s, w], 0),
    (S([P((3, 2), s, Obj('soup', (3, 2), ('onion', 3, 5))),P((2, 2), w, Obj('onion', (2, 2)))],{}, order_list=['onion', 'any']), [interact, n], delivery_reward),
    (S([P((3, 2), s),P((2, 1), n, Obj('onion', (2, 1)))],{}, order_list=['any']), [e, interact], 0),
    (S([P((3, 2), e),P((2, 1), n)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=['any']), [interact, s], 0),
    (S([P((3, 2), e, Obj('tomato', (3, 2))),P((2, 2), s)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},order_list=['any']), [w, w], 0),
    (S([P((2, 2), w, Obj('tomato', (2, 2))),P((1, 2), w)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},order_list=['any']), [n, interact], 0),
    (S([P((2, 1), n, Obj('tomato', (2, 1))),P((1, 2), w, Obj('tomato', (1, 2)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},order_list=['any']), [interact, interact], 0),
    (S([P((2, 1), n, Obj('tomato', (2, 1))),P((1, 2), w, Obj('tomato', (1, 2)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},order_list=['any']), [s, interact], 0),
    (S([P((2, 2), s, Obj('tomato', (2, 2))),P((1, 2), w, Obj('tomato', (1, 2)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},order_list=['any']), [interact, interact], 0),
    (S([P((2, 2), s),P((1, 2), w, Obj('tomato', (1, 2)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0)),(2, 3): Obj('soup', (2, 3), ('tomato', 1, 0))}, order_list=['any']), [interact, interact], 0)
]

curr_ep_rewards = [s_a_r[2] for s_a_r in s_a_r_pairs]

traj = {
    "ep_observations": [[s_a_r[0] for s_a_r in s_a_r_pairs]],
    "ep_actions": [[s_a_r[1] for s_a_r in s_a_r_pairs]],
    "ep_rewards": [curr_ep_rewards],
    "ep_dones": [False] * len(s_a_r_pairs),

    "ep_returns": [sum(curr_ep_rewards)],
    "ep_returns_sparse": [sum(curr_ep_rewards)],
    "ep_lengths": [len(curr_ep_rewards)],
    "mdp_params": [{
        "layout_name": "mdp_test",
        "cook_time": 5,
        "start_order_list": ["onion", "any"],
        "num_items_for_soup": 3,
        "rew_shaping_params": None
    }],
    "env_params": [{
        "horizon": 100,
        "start_state_fn": None
    }]
}

AgentEvaluator.save_traj_as_json(traj, "test_traj")
