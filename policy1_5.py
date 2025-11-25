import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralizedPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        num_robots,
        num_move,
        num_offload,
        num_partner,
        num_meeting,
        num_amount,
        hidden_dim=256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_robots = num_robots
        self.num_move = num_move
        self.num_offload = num_offload
        self.num_partner = num_partner
        self.num_meeting = num_meeting
        self.num_amount = num_amount

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.move_head = nn.Linear(hidden_dim, num_robots * num_move)
        self.offload_head = nn.Linear(hidden_dim, num_robots * num_offload)
        self.partner_head = nn.Linear(hidden_dim, num_robots * num_partner)
        self.meeting_head = nn.Linear(hidden_dim, num_robots * num_meeting)
        self.amount_head = nn.Linear(hidden_dim, num_robots * num_amount)

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, masks=None):
        x = self.trunk(obs)
        B = x.size(0)
        def reshape(head, dim):
            return head.view(B, self.num_robots, dim)
        move_logits = reshape(self.move_head(x), self.num_move)
        offload_logits = reshape(self.offload_head(x), self.num_offload)
        partner_logits = reshape(self.partner_head(x), self.num_partner)
        meeting_logits = reshape(self.meeting_head(x), self.num_meeting)
        amount_logits = reshape(self.amount_head(x), self.num_amount)

        if masks is not None:
            big_neg = -1e9
            if "movement" in masks and masks["movement"] is not None:
                move_logits = move_logits + (1.0 - masks["movement"]) * big_neg
            if "partner" in masks and masks["partner"] is not None:
                partner_logits = partner_logits + (1.0 - masks["partner"]) * big_neg
            if "meeting" in masks and masks["meeting"] is not None:
                meeting_logits = meeting_logits + (1.0 - masks["meeting"]) * big_neg
            if "amount" in masks and masks["amount"] is not None:
                amount_logits = amount_logits + (1.0 - masks["amount"]) * big_neg

        value = self.value_head(x).squeeze(-1)
        return {
            "move_logits": move_logits,
            "offload_logits": offload_logits,
            "partner_logits": partner_logits,
            "meeting_logits": meeting_logits,
            "amount_logits": amount_logits,
            "value": value,
        }

    def _sample_head(self, logits):
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, ent

    def sample_action(self, obs, masks=None):
        out = self.forward(obs, masks)
        move_logits = out["move_logits"]
        offload_logits = out["offload_logits"]
        partner_logits = out["partner_logits"]
        meeting_logits = out["meeting_logits"]
        amount_logits = out["amount_logits"]
        value = out["value"]

        move, lp_m, ent_m = self._sample_head(move_logits)
        offload, lp_o, ent_o = self._sample_head(offload_logits)
        partner, lp_p, ent_p = self._sample_head(partner_logits)
        meeting, lp_M, ent_M = self._sample_head(meeting_logits)
        amount, lp_w, ent_w = self._sample_head(amount_logits)

        logp = lp_m.sum(dim=1) + lp_o.sum(dim=1) + lp_p.sum(dim=1) + lp_M.sum(dim=1) + lp_w.sum(dim=1)
        entropy = ent_m.sum(dim=1) + ent_o.sum(dim=1) + ent_p.sum(dim=1) + ent_M.sum(dim=1) + ent_w.sum(dim=1)

        action = {
            "move": move.squeeze(0),
            "offload": offload.squeeze(0),
            "partner": partner.squeeze(0),
            "meeting": meeting.squeeze(0),
            "amount": amount.squeeze(0),
        }
        return action, logp.squeeze(0), entropy.squeeze(0), value.squeeze(0)

    def evaluate_actions(self, obs, action, masks=None):
        out = self.forward(obs, masks)
        move_logits = out["move_logits"]
        offload_logits = out["offload_logits"]
        partner_logits = out["partner_logits"]
        meeting_logits = out["meeting_logits"]
        amount_logits = out["amount_logits"]
        value = out["value"]

        def cat_stats(logits, a):
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            logp = dist.log_prob(a)
            ent = dist.entropy()
            return logp.sum(dim=1), ent.sum(dim=1)

        lp_m, ent_m = cat_stats(move_logits, action["move"])
        lp_o, ent_o = cat_stats(offload_logits, action["offload"])
        lp_p, ent_p = cat_stats(partner_logits, action["partner"])
        lp_M, ent_M = cat_stats(meeting_logits, action["meeting"])
        lp_w, ent_w = cat_stats(amount_logits, action["amount"])

        logp = lp_m + lp_o + lp_p + lp_M + lp_w
        entropy = ent_m + ent_o + ent_p + ent_M + ent_w

        return logp, entropy, value
