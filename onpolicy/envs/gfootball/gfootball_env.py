
"""
    MARL environment for google football
"""

import gfootball.env as env
from gfootball.env.observation_preprocessing import SMM_WIDTH
from gfootball.env.observation_preprocessing import SMM_HEIGHT


class MultiAgentEnv:
    def seed(self, seed):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class GoogleFootballEnv(MultiAgentEnv):

    """Initialize the environment

  Args:
    num_of_left_agents: Number of left players an agent
        controls.
    num_of_right_agents: Number of right players an agent
        controls.
    env_name: a name of a scenario to run, e.g. "11_vs_11_stochastic".
      The list of scenarios can be found in directory "scenarios".
    stacked: If True, stack 4 observations, otherwise, only the last
      observation is returned by the environment.
      Stacking is only possible when representation is one of the following:
      "pixels", "pixels_gray" or "extracted".
      In that case, the stacking is done along the last (i.e. channel)
      dimension.
    representation: String to define the representation used to build
      the observation. It can be one of the following:
      'pixels': the observation is the rendered view of the football field
        downsampled to 'channel_dimensions'. The observation size is:
        'channel_dimensions'x3 (or 'channel_dimensions'x12 when "stacked" is
        True).
      'pixels_gray': the observation is the rendered view of the football field
        in gray scale and downsampled to 'channel_dimensions'. The observation
        size is 'channel_dimensions'x1 (or 'channel_dimensions'x4 when stacked
        is True).
      'extracted': also referred to as super minimap. The observation is
        composed of 4 planes of size 'channel_dimensions'.
        Its size is then 'channel_dimensions'x4 (or 'channel_dimensions'x16 when
        stacked is True).
        The first plane P holds the position of players on the left
        team, P[y,x] is 255 if there is a player at position (x,y), otherwise,
        its value is 0.
        The second plane holds in the same way the position of players
        on the right team.
        The third plane holds the position of the ball.
        The last plane holds the active player.
      'simple115'/'simple115v2': the observation is a vector of size 115.
        It holds:
         - the ball_position and the ball_direction as (x,y,z)
         - one hot encoding of who controls the ball.
           [1, 0, 0]: nobody, [0, 1, 0]: left team, [0, 0, 1]: right team.
         - one hot encoding of size 11 to indicate who is the active player
           in the left team.
         - 11 (x,y) positions for each player of the left team.
         - 11 (x,y) motion vectors for each player of the left team.
         - 11 (x,y) positions for each player of the right team.
         - 11 (x,y) motion vectors for each player of the right team.
         - one hot encoding of the game mode. Vector of size 7 with the
           following meaning:
           {NormalMode, KickOffMode, GoalKickMode, FreeKickMode,
            CornerMode, ThrowInMode, PenaltyMode}.
         Can only be used when the scenario is a flavor of normal game
         (i.e. 11 versus 11 players).
    rewards: Comma separated list of rewards to be added.
       Currently supported rewards are 'scoring' and 'checkpoints'.
    write_goal_dumps: whether to dump traces up to 200 frames before goals.
    write_full_episode_dumps: whether to dump traces for every episode.
    render: whether to render game frames.
       Must be enable when rendering videos or when using pixels
       representation.
    write_video: whether to dump videos when a trace is dumped.
    dump_frequency: how often to write dumps/videos (in terms of # of episodes)
      Sub-sample the episodes for which we dump videos to save some disk space.
    logdir: directory holding the logs.
    extra_players: A list of extra players to use in the environment.
        Each player is defined by a string like:
        "$player_name:left_players=?,right_players=?,$param1=?,$param2=?...."
    channel_dimensions: (width, height) tuple that represents the dimensions of
       SMM or pixels representation.
    other_config_options: dict that allows directly setting other options in
       the Config

"""

    def __init__(self, num_of_left_agents,
                 num_of_right_agents=0,
                 env_name="test_example_multiagent",
                 stacked=False,
                 representation='extracted',
                 rewards='scoring',
                 write_goal_dumps=False,
                 write_full_episode_dumps=False,
                 render=False,
                 write_video=False,
                 dump_frequency=1,
                 extra_players=None,
                 channel_dimensions=(
                     SMM_WIDTH,
                     SMM_HEIGHT),
                 other_config_options={}) -> None:

        self.env = env.create_environment(
            env_name=env_name,
            stacked=stacked,
            representation=representation,
            rewards=rewards,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=render,
            write_video=write_video,
            dump_frequency=dump_frequency,
            logdir="/tmp/gfootball_log",
            extra_players=extra_players,
            number_of_left_players_agent_controls=num_of_left_agents,
            number_of_right_players_agent_controls=num_of_right_agents,
            channel_dimensions=channel_dimensions,
            other_config_options=other_config_options
        )


if __name__ == "__main__":
    e = GoogleFootballEnv(1, env_name="1_vs_1_easy")
