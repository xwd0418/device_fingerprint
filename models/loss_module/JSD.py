import torch, torch.nn as nn
import torch.nn.functional as F
class JSD(nn.Module):
    # @torch.no_grad()
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = F.softmax(p.view(-1, p.size(-1)), dim=1), F.softmax(q.view(-1, q.size(-1)), dim=1)
        m = (0.5 * (p + q)).log()
        jsd = 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
        return jsd