from typing import Tuple, List
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG_INF = -float("inf")


class CTCLossWrapper(nn.Module):

    def __init__(self, blank_idx: int = 0, reduction: str = "mean"):
        super().__init__()
        self.blank_idx = blank_idx
        self.reduction = reduction

    def forward(self,
                feature: torch.Tensor,
                targets: torch.Tensor,
                feature_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        :param feature:             (batch_size, feat_len, vocab_size)
        :param targets:             (batch_size, target_len)        long
        :param feature_lengths:     (batch_size,)                   long
        :param target_lengths:      (batch_size,)                   long
        :return:
                loss                (1,)
        """
        log_probs = F.log_softmax(feature, dim=-1)
        # ctc loss require time-first
        log_probs = log_probs.transpose(0, 1).contiguous()  # (s, b, v)

        loss = F.ctc_loss(log_probs, targets, feature_lengths, target_lengths,
                          blank=self.blank_idx, reduction=self.reduction)
        return loss


class CTCDecoder(nn.Module):

    def __init__(self, blank_idx: int = 0, vocab_size: int = 29):
        super().__init__()
        self.blank_idx = blank_idx
        self.vocab_size = vocab_size

    @torch.no_grad()
    def forward(self,
                feature: torch.Tensor,
                lengths: torch.Tensor,
                beam_width: int = 1) -> Tuple[List[List[int]], List[float]]:
        """
        :param feature:         (batch_size, feat_len, vocab_size)
        :param lengths:         (batch_size,)                   long
        :param beam_width:      int
        :return:
                prediction:     list of integers
                logp_score:     float
        """
        if beam_width <= 1:
            return self.greedy_decode(feature, lengths)
        else:
            beam_width = min(beam_width, self.vocab_size)
            return self.beamsearch_decode(feature, lengths, beam_width)

    def greedy_decode(self,
                      feature: torch.Tensor,
                      lengths: torch.Tensor) -> Tuple[List[List[int]], List[float]]:
        # ---------------------------------------------------------------- #
        batch_size, _, vocab_size = feature.shape
        assert lengths.shape[0] == batch_size

        prediction = [[] for _ in range(batch_size)]
        logp_scores = [0.0 for _ in range(batch_size)]

        # ---------------------------------------------------------------- #
        feature = F.log_softmax(feature.float(), dim=-1)

        # per-sample
        for batch_idx in range(batch_size):
            length = lengths[batch_idx].item()
            feat = feature[batch_idx, :length]  # (s, V)
            max_scores, max_tokens = torch.max(feat, dim=-1)  # (s,), (s,)

            logp_scores[batch_idx] = torch.sum(max_scores).item()

            tokens = max_tokens.tolist()
            current_token = self.blank_idx
            for t in tokens:
                if ((t != current_token) or (current_token == self.blank_idx)) and (t != self.blank_idx):
                    prediction[batch_idx].append(t)
                current_token = t

        return prediction, logp_scores

    def beamsearch_decode(self,
                          feature: torch.Tensor,
                          lengths: torch.Tensor,
                          beam_width: int) -> Tuple[List[List[int]], List[float]]:
        # ---------------------------------------------------------------- #
        batch_size, _, vocab_size = feature.shape
        assert lengths.shape[0] == batch_size

        prediction = [[] for _ in range(batch_size)]
        logp_scores = [0.0 for _ in range(batch_size)]

        # ---------------------------------------------------------------- #
        feature = F.log_softmax(feature.float(), dim=-1)

        def logsumexp(*values) -> float:
            res = torch.logsumexp(torch.tensor(values), dim=0)
            return res.item()

        # per-sample
        for batch_idx in range(batch_size):
            length = lengths[batch_idx].item()
            feat = feature[batch_idx, :length]  # (s, V)

            # initial beam prob.
            beam = [(tuple(), (0.0, _NEG_INF))]  # prefix, (p_blank, p_non_blank)

            for t in range(length):
                next_beam = defaultdict(lambda: (_NEG_INF, _NEG_INF))

                for v in range(vocab_size):
                    p = feat[t, v].item()

                    for prefix, (p_b, p_nb) in beam:
                        if v == self.blank_idx:
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                            next_beam[prefix] = (n_p_b, n_p_nb)
                            continue

                        end_t = prefix[-1] if prefix else None
                        n_prefix = prefix + (v,)
                        n_p_b, n_p_nb = next_beam[n_prefix]
                        if v != end_t:
                            n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                        else:
                            n_p_nb = logsumexp(n_p_nb, p_b + p)

                        # we add LM score here
                        next_beam[n_prefix] = (n_p_b, n_p_nb)

                        if v == end_t:
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_nb = logsumexp(n_p_nb, p_nb + p)
                            next_beam[prefix] = (n_p_b, n_p_nb)

                beam = sorted(next_beam.items(),
                              key=lambda x: logsumexp(x[1][0], x[1][1]),
                              reverse=True)
                beam = beam[:beam_width]

            best_beam = beam[0]
            prediction[batch_idx] = list(best_beam[0])
            logp_scores[batch_idx] = logsumexp(best_beam[1][0], best_beam[1][1])

        return prediction, logp_scores
