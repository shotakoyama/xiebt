import torch
from fairseq import options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from .task import BackTranslationTask
from .util import *

def xiebt_generate(cfg):
    logger = make_logger()
    logger.info(cfg)
    fix_seed(cfg)
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    task = BackTranslationTask.setup_task(cfg.task)
    task.load_dataset(cfg.dataset.gen_subset)
    src_dict, tgt_dict = load_dicts(task)
    models = load_models(logger, cfg, task, use_cuda)
    progress = make_progress_bar(cfg, task, models)
    generator = task.build_generator(models, cfg)

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        hypos = task.inference_step(generator, models, sample)

        for i, sample_id in enumerate(sample["id"].tolist()):
            print_hypos(cfg, sample, hypos, generator, i, sample_id, src_dict, tgt_dict)


def main():
    parser = options.get_generation_parser()
    parser.add_argument('--beta-random', default = 8.0, type = float)
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    xiebt_generate(cfg)


