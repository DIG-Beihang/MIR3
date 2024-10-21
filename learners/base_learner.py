class BaseLearner:
    def collect(self, batch, t, global_t, actor_hs=None):
        pass

    def learn(self, sample, episodes):
        raise NotImplementedError