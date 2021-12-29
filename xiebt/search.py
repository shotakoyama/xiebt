import torch
from fairseq.search import Search

class RandomNoisingBeamSearch(Search):

    def __init__(self, tgt_dict, beta_random):
        super().__init__(tgt_dict)
        self.constraint_states = None
        self.beta_random = beta_random

    @torch.jit.export
    def step(
            self,
            step: int,
            lprobs,
            scores,
            prev_output_tokens = None,
            original_batch_idxs = None):

        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        # Xie Random Noising
        # In Xie+, Kiyono+, Koyama+, "score" means sum of log-probabilities.
        # This code adds noise to log probability.
        # But, this is same to adding noise to score.
        lprobs += self.beta_random * torch.rand_like(lprobs)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf

