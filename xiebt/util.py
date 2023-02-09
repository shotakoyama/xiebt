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


def fix_seed(cfg):
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)


def load_dicts(task):
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary
    return src_dict, tgt_dict


def load_models(logger, cfg, task, use_cuda):
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, save_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        task = task,
        suffix = cfg.checkpoint.checkpoint_suffix,
        strict = (cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards = cfg.checkpoint.checkpoint_shard_count)

    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)
    return models


def make_progress_bar(cfg, task, models):
    itr = task.get_batch_iterator(
        dataset = task.dataset(cfg.dataset.gen_subset),
        max_tokens = cfg.dataset.max_tokens,
        max_sentences = cfg.dataset.batch_size,
        max_positions = utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]),
        ignore_invalid_inputs = cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple = cfg.dataset.required_batch_size_multiple,
        num_shards = cfg.distributed_training.distributed_world_size,
        shard_id = cfg.distributed_training.distributed_rank,
        num_workers = cfg.dataset.num_workers,
        data_buffer_size = cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle = False)
    progress = progress_bar.progress_bar(
        itr,
        log_format = cfg.common.log_format,
        log_interval = cfg.common.log_interval,
        default_log_format = ("tqdm" if not cfg.common.no_progress_bar else "simple"))
    return progress


def print_hypos(cfg, sample, hypos, generator, i, sample_id, src_dict, tgt_dict):
    src_tokens = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], tgt_dict.pad())
    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
    print("S-{}\t{}".format(sample_id, src_str))

    # Process top predictions
    for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
            hypo_tokens = hypo["tokens"].int().cpu(),
            src_str = src_str,
            alignment = hypo["alignment"],
            align_dict = None,
            tgt_dict = tgt_dict,
            remove_bpe = cfg.common_eval.post_process,
            extra_symbols_to_ignore = {generator.eos})
        score = hypo["score"] / math.log(2)  # convert to base 2
        # original hypothesis (after tokenization and BPE)
        print("H-{}\t{}\t{}".format(sample_id, score, hypo_str))

