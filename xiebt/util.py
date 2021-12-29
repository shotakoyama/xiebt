import os
import sys
import math
import logging
import numpy as np
from fairseq import checkpoint_utils, utils
from fairseq.logging import progress_bar

def make_logger():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout)
    logger = logging.getLogger("fairseq_cli.generate")
    return logger


def fix_seed(args):
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)


def load_dicts(task):
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary
    return src_dict, tgt_dict


def load_models(logger, args, task, use_cuda):
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count)

    for model in models:
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)
    return models


def make_progress_bar(args, task, models):
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"))
    return progress


def print_hypos(args, sample, hypos, generator, i, sample_id, src_dict, tgt_dict):
    src_tokens = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], tgt_dict.pad())
    src_str = src_dict.string(src_tokens, args.remove_bpe)
    print("S-{}\t{}".format(sample_id, src_str))

    # Process top predictions
    for j, hypo in enumerate(hypos[i][: args.nbest]):
        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
            hypo_tokens=hypo["tokens"].int().cpu(),
            src_str=src_str,
            alignment=hypo["alignment"],
            align_dict=None,
            tgt_dict=tgt_dict,
            remove_bpe=args.remove_bpe,
            extra_symbols_to_ignore = {generator.eos})
        score = hypo["score"] / math.log(2)  # convert to base 2
        # original hypothesis (after tokenization and BPE)
        print("H-{}\t{}\t{}".format(sample_id, score, hypo_str))

