class BaseScriptPeriod:
    """A base class for script period.
    """
    def __init__(self, period_name):
        self.period_name = period_name
    
    def reset(self, mdp, state, player_idx):
        """reset some script period
        """
        raise NotImplementedError
    
    def step(self, mdp, state, player_idx):
        raise NotImplementedError
    
    def done(self, mdp, state, player_idx):
        raise NotImplementedError

class BaseScriptAgent:
    """A script agent consists of several script periods.
    """
    def __init__(self):
        pass

    def reset(self, mdp, state, player_idx):
        """reset state
        """
        pass

    def step(self, mdp, state, player_idx):
        raise NotImplementedError
