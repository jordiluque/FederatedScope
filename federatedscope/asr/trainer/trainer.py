import torch
import logging
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer
from federatedscope.core.trainers import BaseTrainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.asr.model.adapter_builder import AdapterModel
from federatedscope.asr.dataloader.dataloader import ASRDataCollator, get_asr_tokenizer_extractor
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-Maltese-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


class ASRTrainer(BaseTrainer):
    def __init__(self, model, data, device, **kwargs):
        self.training_args = Seq2SeqTrainingArguments(
                output_dir="temp",  # change to a repo name of your choice
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
                learning_rate=1e-3,
                warmup_steps=50,
                num_train_epochs=3,
                eval_strategy="no",
                fp16=True,
                per_device_eval_batch_size=8,
                generation_max_length=128,
                logging_steps=25,
                remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
                label_names=["labels"],  # same reason as above
        )
        
        # NN modules
        self.model = model
        # FS `ClientData` or your own data
        
        self.data = data
        #print("Len train2: ",len(self.data['train']))
        #print(self.data['train'].dataset)
        #train_features, train_labels = next(iter(data['train']))
        #print (train_labels[:100])
        #self.input_features = common_voice[:100]['input_features']
        #self.labels = common_voice[:100]['labels']
        
        # Device name
        self.device = device
        # kwargs
        self.kwargs = kwargs
         
        self.tokenizer, self.feature_extractor, self.processor = \
                    get_asr_tokenizer_extractor( "openai/whisper-large-v2", "data/", "Maltese",
                    "transcribe")
        self.data_collator = ASRDataCollator(processor=self.processor)
        # Criterion & Optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
            

    def train(self):
         
        self.trainer = Seq2SeqTrainer(
                args=self.training_args,
                model=self.model,
                train_dataset=self.data['train'].dataset,
                #eval_dataset=self.data['val'],
                data_collator=self.data_collator,
                # compute_metrics=compute_metrics,
                tokenizer=self.processor.feature_extractor,
                #callbacks=[SavePeftModelCallback],
        )
        '''
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=0.001,
                                         momentum=0.9,
                                         weight_decay=1e-4)
        '''
        # _hook_on_fit_start_init
        #self.model.to(self.device)
        self.trainer.train()

        total_loss = num_samples = 0
        # _hook_on_batch_start_init
        for x, y in self.data['train']:
            # _hook_on_batch_forward
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            # _hook_on_batch_backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # _hook_on_batch_end
            total_loss += loss.item() * y.shape[0]
            num_samples += y.shape[0]

        # _hook_on_fit_end
        return num_samples, self.model.cpu().state_dict(), \
            {'loss_total': total_loss, 'avg_loss': total_loss/float(
                num_samples)}

    def evaluate(self, target_data_split_name='test'):
        import torch
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            total_loss = num_samples = 0
            # _hook_on_batch_start_init
            for x, y in self.data[target_data_split_name]:
                # _hook_on_batch_forward
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)

                # _hook_on_batch_end
                total_loss += loss.item() * y.shape[0]
                num_samples += y.shape[0]

            # _hook_on_fit_end
            return {
                f'{target_data_split_name}_loss': total_loss,
                f'{target_data_split_name}_total': num_samples,
                f'{target_data_split_name}_avg_loss': total_loss /
                float(num_samples)
            }

    def update(self, model_parameters, strict=False):
        self.model.load_state_dict(model_parameters, strict)

    def get_model_para(self):
        return self.model.cpu().state_dict()



class KKTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.asr.deepspeed.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.asr.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.asr.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):
        print(len(ctx.data_batch['input_features']))
        input_ids = ctx.data_batch['input_features'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        #attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        if ctx.cfg.asr.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels)
                                       #attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels)
                                #attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.asr.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            ctx.optimizer.zero_grad()
            ctx.loss_task.backward()

            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)

            ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.asr.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        setattr(ctx, 'eval_metrics', eval_results)

        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.asr.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_features'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                #attention_mask = ctx.data_batch['attention_mask'].to(
                    #ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=input_ids.total())
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids).total())
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_asr_trainer(trainer_type):
    if trainer_type == 'asrtrainer':
        trainer_builder = ASRTrainer
        return trainer_builder


register_trainer('asrtrainer', call_asr_trainer)
