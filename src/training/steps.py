class Steps:
    def __init__(self, total=0, current_bs=0):
        self.total = total
        self.current_bs = current_bs

    def step(self):
        self.total += 1
        self.current_bs += 1

    def reset_current_bs(self):
        self.current_bs = 0

    def sfo(self, batch_size: int):
        return self.current_bs * batch_size

    def state_dict(self):
        return {"total": self.total, "current_bs": self.current_bs}

    def load_state_dict(self, state_dict):
        self.total = state_dict["total"]
        self.current_bs = state_dict["current_bs"]
