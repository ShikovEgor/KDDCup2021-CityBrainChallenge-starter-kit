import torch as th
from torch.optim import Adam

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
        for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

class Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.optimiser = Adam(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)


    def train_batch(self, obs, rwd):
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def cuda(self):
        self.mac.cuda()


    # def save_models(self, path):
    #     self.mac.save_models(path)
    #     if self.mixer is not None:
    #         th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
    #     th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    # def load_models(self, path):
    #     self.mac.load_models(path)
    #     # Not quite right but I don't want to save target networks
    #     self.target_mac.load_models(path)
    #     if self.mixer is not None:
    #         self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
    #     self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
