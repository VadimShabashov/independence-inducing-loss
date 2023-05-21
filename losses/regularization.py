class L1:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * x.abs().mean()
