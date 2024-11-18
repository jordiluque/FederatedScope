import json
import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_asr_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # ASR related options
    # ---------------------------------------------------------------------- #
    cfg.asr = CN()

    # ---------------------------------------------------------------------- #
    # Cache for asr
    # ---------------------------------------------------------------------- #
    cfg.asr.cache = CN()
    cfg.asr.cache.model = ''

    # ---------------------------------------------------------------------- #
    # Deepspeed related options
    # ---------------------------------------------------------------------- #
    cfg.asr.deepspeed = CN()
    cfg.asr.deepspeed.use = False
    cfg.asr.deepspeed.ds_config = ''

    # ---------------------------------------------------------------------- #
    # Adapters for asr
    # ---------------------------------------------------------------------- #
    cfg.asr.adapter = CN()
    cfg.asr.adapter.use = False
    cfg.asr.adapter.args = [{}]
    # Move adapter to `cpu` after training, which can save memory but cost
    # more time.
    cfg.asr.adapter.mv_to_cpu = False

    # ---------------------------------------------------------------------- #
    # Offsite-tuning related options
    # ---------------------------------------------------------------------- #
    cfg.asr.offsite_tuning = CN()
    cfg.asr.offsite_tuning.use = False
    cfg.asr.offsite_tuning.strategy = 'drop_layer'
    cfg.asr.offsite_tuning.kwargs = [{}]
    cfg.asr.offsite_tuning.emu_l = 1  # Index of emulator layer left
    cfg.asr.offsite_tuning.emu_r = 10  # Index of emulator layer right

    # Used in `eval`
    cfg.asr.offsite_tuning.eval_type = 'emu'  # Choose one of `[emu, full]`

    # Emulator alignment will use dataset in Server
    cfg.asr.offsite_tuning.emu_align = CN()
    cfg.asr.offsite_tuning.emu_align.use = False
    cfg.asr.offsite_tuning.emu_align.restore_from = ''
    cfg.asr.offsite_tuning.emu_align.save_to = ''
    cfg.asr.offsite_tuning.emu_align.exit_after_align = False

    # Server held-out data
    cfg.asr.offsite_tuning.emu_align.data = CN()
    cfg.asr.offsite_tuning.emu_align.data.root = 'data'
    cfg.asr.offsite_tuning.emu_align.data.type = 'alpaca@asr'
    cfg.asr.offsite_tuning.emu_align.data.splits = [0.8, 0.1, 0.1]

    cfg.asr.offsite_tuning.emu_align.train = CN()
    cfg.asr.offsite_tuning.emu_align.train.local_update_steps = 10
    cfg.asr.offsite_tuning.emu_align.train.batch_or_epoch = 'batch'
    cfg.asr.offsite_tuning.emu_align.train.lm_loss_weight = 0.1
    cfg.asr.offsite_tuning.emu_align.train.kd_loss_weight = 0.9

    cfg.asr.offsite_tuning.emu_align.train.optimizer = CN(new_allowed=True)
    cfg.asr.offsite_tuning.emu_align.train.optimizer.type = 'SGD'
    cfg.asr.offsite_tuning.emu_align.train.optimizer.lr = 0.01


def assert_asr_cfg(cfg):
    if cfg.asr.offsite_tuning.emu_align.use:
        if cfg.asr.offsite_tuning.emu_align.restore_from != '':
            logger.warning(
                'Enabling `restore_from` in offsite_tuning emulator '
                'alignment will skip training the emulator.')


register_config("asr", extend_asr_cfg)
