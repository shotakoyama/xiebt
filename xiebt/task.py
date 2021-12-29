from .search import RandomNoisingBeamSearch
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks.translation import TranslationTask

class BackTranslationTask(TranslationTask):

    def build_generator(self, models, args):
        beta_random = getattr(args, 'beta_random', 8.0)
        search_strategy = RandomNoisingBeamSearch(self.target_dictionary, beta_random)
        return SequenceGenerator(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy)

