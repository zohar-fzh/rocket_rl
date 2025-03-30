
class Memory:
    def __init__(self):
        self.actions = []   # 行动(共4种)
        self.states = []    # 状态, 由8个数字组成
        self.logprobs = []  # 概率
        self.rewards = []   # 奖励
        self.is_dones = []  ## 游戏是否结束 is_terminals?
        return

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_dones[:]
        return