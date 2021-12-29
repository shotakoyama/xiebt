import torch
from fairseq import options, utils
from .task import BackTranslationTask
from .util import *

def xiebt_generate(args):
    logger = make_logger()
    logger.info(args)
    fix_seed(args)
    use_cuda = torch.cuda.is_available() and not args.cpu
    task = BackTranslationTask.setup_task(args)
    task.load_dataset(args.gen_subset)
    src_dict, tgt_dict = load_dicts(task)
    models = load_models(logger, args, task, use_cuda)
    progress = make_progress_bar(args, task, models)
    generator = task.build_generator(models, args)

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        hypos = task.inference_step(generator, models, sample)

        for i, sample_id in enumerate(sample["id"].tolist()):
            print_hypos(args, sample, hypos, generator, i, sample_id, src_dict, tgt_dict)


def main():
    parser = options.get_generation_parser()
    parser.add_argument('--beta-random', default = 8.0, type = float)
    args = options.parse_args_and_arch(parser)
    xiebt_generate(args)


