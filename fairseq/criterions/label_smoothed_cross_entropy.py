# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import numpy as np
import random
from fairseq import checkpoint_utils


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    # import pdb; pdb.set_trace()
    # print(lprobs.shape, target.shape)
    # print(ignore_index)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def l2_att_loss(attn_all, attn_all_ref, lprobs, target, ignore_index=None, reduce=True):
    # loss = torch.nn.MSELoss()
    # import pdb; pdb.set_trace()
    # return torch.nn.functional.mse_loss(torch.cat(attn_all,dim=0), torch.cat(attn_all_ref,dim=0))

    attn_all = [a.float() for a in attn_all]
    attn_all_ref = [a.float() for a in attn_all_ref]

    attn_loss = torch.nn.functional.mse_loss(torch.cat(attn_all,dim=0), torch.cat(attn_all_ref,dim=0), reduction='none') # L*H, B, Tout, Tin
    attn_loss = attn_loss.mean(0).mean(-1).view(-1).unsqueeze(1) # B*Tout, 1
    # print(attn_loss.shape)

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        # if pad_mask.any(): 
        #     print(pad_mask[-5:], attn_loss[-5:])
        #     import pdb; pdb.set_trace()
        attn_loss.masked_fill_(pad_mask, 0.0)
    else:
        attn_loss = attn_loss.squeeze(-1)
    if reduce:
        attn_loss = attn_loss.sum()
    # pdb.set_trace()
    # print(attn_loss.shape, attn_loss)
    return attn_loss




@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        # import pdb; pdb.set_trace()
        # print(lprobs.shape, target.shape)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


# ------------------------------------

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig_fr(LabelSmoothedCrossEntropyCriterionConfig):
    nb_iters: int = field(
        default=1,
        metadata={"help": "nb of iterations when producing generated back history"},
    )

@register_criterion(
    "label_smoothed_cross_entropy_fr", dataclass=LabelSmoothedCrossEntropyCriterionConfig_fr
)
class LabelSmoothedCrossEntropyCriterion_fr(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        nb_iters=1,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size=ignore_prefix_size, report_accuracy=report_accuracy)
        self.nb_iters = nb_iters

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import pdb; pdb.set_trace()
        # print(type(sample['net_input']['prev_output_tokens']))
        # print(sample['net_input']['prev_output_tokens'][0])
        for k in range(self.nb_iters):
            with torch.no_grad():
                net_output = model(**sample["net_input"])
                sample['net_input']['prev_output_tokens'][:,(k+1):] = torch.topk(net_output[0],1)[1].squeeze()[:,:-(k+1)]
            # print(sample['net_input']['prev_output_tokens'][0])
            net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig_ss(LabelSmoothedCrossEntropyCriterionConfig_fr):
    ss_max: float = field(
        default=0.5,
        metadata={"help": "probability of using generated back history"},
    )
    ss_step_start: int = field(
        default=10000,
        metadata={"help": "nb of steps when ss begins"},
    )
    ss_step_peak: int = field(
        default=20000,
        metadata={"help": "nb of steps when ss peaks"},
    )

@register_criterion(
    "label_smoothed_cross_entropy_ss", dataclass=LabelSmoothedCrossEntropyCriterionConfig_ss
)
class LabelSmoothedCrossEntropyCriterion_ss(LabelSmoothedCrossEntropyCriterion_fr):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        nb_iters=1,
        ss_max=0.5,
        ss_step_start=10000,
        ss_step_peak=20000,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size=ignore_prefix_size, report_accuracy=report_accuracy, nb_iters=nb_iters)
        self.ss_max = ss_max
        self.ss_step_start = ss_step_start
        self.ss_step_peak = ss_step_peak
        self.cnt_steps = 0

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import pdb; pdb.set_trace()

        # scheduled sampling
        # progress = max(1.0, 1.0 * 2 * step / total_steps) # wrong legacy implementation
        self.cnt_steps += 1
        progress = float(self.cnt_steps - self.ss_step_start) / float(self.ss_step_peak - self.ss_step_start)
        progress = np.clip(progress, 0.0, 1.0)

        # linear schedule
        teacher_forcing_ratio = 1.0 - self.ss_max * progress

        # Inverse sigmoid decay
        # ss_k = 10 # 5 10
        # teacher_forcing_ratio = 1.0 - self.ss_max + self.ss_max * (ss_k / (ss_k + np.exp(progress * 100 / ss_k)))

        for k in range(self.nb_iters):
            if random.random() > teacher_forcing_ratio: # use generated back history
                with torch.no_grad():
                    net_output = model(**sample["net_input"])
                    sample['net_input']['prev_output_tokens'][:,(k+1):] = torch.topk(net_output[0],1)[1].squeeze()[:,:-(k+1)]
            net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig_af(LabelSmoothedCrossEntropyCriterionConfig):
    model_tf_path: str = field(
        default="",
        metadata={"help": "path of the tf model"},
    )
    scale_attn_loss: float = field(
        default=1.0 * 1e3,
        metadata={"help": "path of the tf model"},
    )
    no_af_mask: bool = field(
        default=False,
        metadata={"help": "mask af loss, in the same way as tf loss"},
    )
    # avg_attn: bool = field(
    #     default=False,
    #     metadata={"help": "average ref attn heads, force only one head"},
    # )

@register_criterion(
    "label_smoothed_cross_entropy_af_ref", dataclass=LabelSmoothedCrossEntropyCriterionConfig_af
)
class LabelSmoothedCrossEntropyCriterion_af_ref(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        model_tf_path=None,
        scale_attn_loss=1.0,
        no_af_mask=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size=ignore_prefix_size, report_accuracy=report_accuracy)
        self.model_tf_path = model_tf_path
        self.scale_attn_loss = scale_attn_loss
        self.no_af_mask = no_af_mask
        # # import pdb; pdb.set_trace()
        # # ckpt = torch.load(model_tf_path)
        # # self.model_tf = ckpt['model']
        # models, cfg = checkpoint_utils.load_model_ensemble(
        #     utils.split_paths(model_tf_path),
        #     task=task,
        # )
        # # print(cfg)
        # # pdb.set_trace()

        # use_cuda = torch.cuda.is_available() and not cfg.common.cpu
        # # Optimize model for generation
        # model = models[0]
        # if cfg.common.fp16:
        #     model.half()
        # if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        #     model.cuda()
        # model.prepare_for_inference_(cfg)
        # self.model_tf = model

    def forward(self, model, model_tf, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import pdb; pdb.set_trace()

        # use model_tf to get align_ref in tf mode
        with torch.no_grad():
            net_output_tf = model_tf(**sample["net_input"])

        # pass align_ref, while still using reference back history
        sample['net_input']['attn_all_ref'] = [ x.detach() for x in net_output_tf[1]['attn_all'] ]
        # sample['net_input']['prev_output_tokens'][:,1:] = torch.topk(net_output_tf[0],1)[1].squeeze()[:,:-1].detach()
        net_output = model(**sample["net_input"])

        loss, nll_loss, att_loss = self.compute_loss_af(model, net_output, sample, reduce=reduce)
        # print(loss, nll_loss, att_loss)

        # attn_all = torch.stack(net_output_tf[1]['attn_all'])
        # dirFile = '/home/dawna/tts/qd212/models/fairseq/checkpoints/en_de_wmt16/0013_transformer_af_ref_scale1.0/attn_all_ref.npy'
        # pdb.set_trace()
        # np.save(dirFile, np.array(attn_all.permute(2,0,1,3,4).cpu()))

        # attn_all = torch.stack(net_output[1]['attn_all'])
        # dirFile = '/home/dawna/tts/qd212/models/fairseq/checkpoints/en_de_wmt16/0013_transformer_af_ref_scale1.0/attn_all_gen.npy'
        # pdb.set_trace()
        # np.save(dirFile, np.array(attn_all.permute(2,0,1,3,4).detach().cpu()))

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "att_loss": att_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if 'attn_all_ref' in sample['net_input']: del sample['net_input']['attn_all_ref']
        return loss, sample_size, logging_output

    def compute_loss_af(self, model, net_output, sample, reduce=True):
        # import pdb; pdb.set_trace()
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        tmp, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        att_loss = l2_att_loss(
            net_output[1]['attn_all'], 
            sample['net_input']['attn_all_ref'],
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduce=reduce,
        ) * self.scale_attn_loss
        loss = tmp + att_loss
        return loss, nll_loss, att_loss

    def compute_loss_af_bkup_nomask(self, model, net_output, sample, reduce=True):
        # import pdb; pdb.set_trace()
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        tmp, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        ignore_index_af = None if self.no_af_mask else self.padding_idx
        att_loss = l2_att_loss(
            net_output[1]['attn_all'], 
            sample['net_input']['attn_all_ref'],
            lprobs,
            target,
            ignore_index=ignore_index_af,
            reduce=reduce,
        ) * self.scale_attn_loss
        loss = tmp + att_loss
        return loss, nll_loss, att_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        attn_loss_sum = sum(log.get("att_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "attn_loss", attn_loss_sum / ntokens, ntokens, round=5
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )


@register_criterion(
    "label_smoothed_cross_entropy_af", dataclass=LabelSmoothedCrossEntropyCriterionConfig_af
)
class LabelSmoothedCrossEntropyCriterion_af(LabelSmoothedCrossEntropyCriterion_af_ref):
    def forward(self, model, model_tf, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import pdb; pdb.set_trace()

        # use model_tf to get align_ref in tf mode
        with torch.no_grad():
            net_output_tf = model_tf(**sample["net_input"])

        # pass align_ref, while still using reference back history
        sample['net_input']['attn_all_ref'] = [ x.detach() for x in net_output_tf[1]['attn_all'] ]
        with torch.no_grad():
            net_output = model(**sample["net_input"])

        sample['net_input']['prev_output_tokens'][:,1:] = torch.topk(net_output[0],1)[1].squeeze()[:,:-1].detach()
        net_output = model(**sample["net_input"])

        # attn_all = torch.stack(net_output_tf[1]['attn_all'])
        # dirFile = '/home/dawna/tts/qd212/models/fairseq/checkpoints/en_de_wmt16/0017_transformer_af_plot_scale1000.0/attn_all_ref.npy'
        # pdb.set_trace()
        # np.save(dirFile, np.array(attn_all.permute(2,0,1,3,4).cpu()))

        # attn_all = torch.stack(net_output[1]['attn_all'])
        # dirFile = '/home/dawna/tts/qd212/models/fairseq/checkpoints/en_de_wmt16/0017_transformer_af_plot_scale1000.0/attn_all_gen.npy'
        # pdb.set_trace()
        # np.save(dirFile, np.array(attn_all.permute(2,0,1,3,4).detach().cpu()))

        # lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # probs=torch.exp(lprobs)
        # tmp = - probs * lprobs
        # print(tmp.sum(-1).mean())

        # lprobs, target = self.get_lprobs_and_target(model_tf, net_output_tf, sample)
        # probs=torch.exp(lprobs)
        # tmp = - probs * lprobs
        # print(tmp.sum(-1).mean())

        loss, nll_loss, att_loss = self.compute_loss_af(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "att_loss": att_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if 'attn_all_ref' in sample['net_input']: del sample['net_input']['attn_all_ref']
        return loss, sample_size, logging_output

    def forward_bkup(self, model, model_tf, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import pdb; pdb.set_trace()

        # use model_tf to get align_ref in tf mode
        with torch.no_grad():
            net_output_tf = model_tf(**sample["net_input"])

        # pass align_ref, while still using reference back history
        sample['net_input']['attn_all_ref'] = [ x.detach() for x in net_output_tf[1]['attn_all'] ]
        sample['net_input']['prev_output_tokens'][:,1:] = torch.topk(net_output_tf[0],1)[1].squeeze()[:,:-1].detach()
        net_output = model(**sample["net_input"])

        loss, nll_loss, att_loss = self.compute_loss_af(model, net_output, sample, reduce=reduce)
        # import pdb; pdb.set_trace()
        # print(self.scale_attn_loss)
        # print(loss, nll_loss, att_loss)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "att_loss": att_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if 'attn_all_ref' in sample['net_input']: del sample['net_input']['attn_all_ref']
        return loss, sample_size, logging_output
        
    def forward_bkup_bkup(self, model, model_tf, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        import pdb; pdb.set_trace()
        print(type(sample['net_input']), sample['net_input'].keys())
        print(sample['target'].shape, sample['net_input']['prev_output_tokens'].shape, sample['net_input']['src_tokens'].shape)

        # use tf to get align_ref
        with torch.no_grad():
            net_output_tf = model_tf(**sample["net_input"])
        # loss_tf, nll_loss_tf = self.compute_loss(model, net_output_tf, sample, reduce=reduce)

        print(net_output_tf[0].shape, net_output_tf[1]['attn'][0].shape, net_output_tf[1]['inner_states'][0].shape)
        print(len(net_output_tf[1]['attn']), len(net_output_tf[1]['inner_states']))
        pdb.set_trace()

        print(len(net_output_tf[1]['attn_all']), net_output_tf[1]['attn_all'][0].shape)

        attn_all = torch.stack(net_output_tf[1]['attn_all'])
        # dirFile = '/home/dawna/tts/qd212/models/fairseq/checkpoints/en_de_wmt16/0012_transformer_fix_tf/attn.npy'
        dirFile = '/home/dawna/tts/qd212/models/fairseq/checkpoints/en_de_wmt16/0000_transformer_af_check/attn_all_ref.npy'
        pdb.set_trace()
        np.save(dirFile, np.array(attn_all.permute(2,0,1,3,4).cpu()))

        sample['net_input']['attn_all_ref'] = [ x.detach() for x in net_output_tf[1]['attn_all'] ]
        sample['net_input']['prev_output_tokens'][:,1:] = torch.topk(net_output_tf[0],1)[1].squeeze()[:,:-1].detach()
        net_output = model(**sample["net_input"])
        loss, nll_loss, att_loss = self.compute_loss_af(model, net_output, sample, reduce=reduce)

        attn_all = torch.stack(net_output[1]['attn_all'])
        # dirFile = '/home/dawna/tts/qd212/models/fairseq/checkpoints/en_de_wmt16/0012_transformer_fix_tf/attn.npy'
        dirFile = '/home/dawna/tts/qd212/models/fairseq/checkpoints/en_de_wmt16/0000_transformer_af_check/attn_all_gen.npy'
        pdb.set_trace()
        np.save(dirFile, np.array(attn_all.permute(2,0,1,3,4).detach().cpu()))

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "att_loss": att_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        del sample['net_input']['attn_all_ref']
        return loss, sample_size, logging_output



@register_criterion(
    "label_smoothed_cross_entropy_af_ref_avg", dataclass=LabelSmoothedCrossEntropyCriterionConfig_af
)
class LabelSmoothedCrossEntropyCriterion_af_ref_avg(LabelSmoothedCrossEntropyCriterion_af_ref):
    def forward(self, model, model_tf, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import pdb; pdb.set_trace()

        # use model_tf to get align_ref in tf mode
        with torch.no_grad():
            net_output_tf = model_tf(**sample["net_input"])

        # pass align_ref, while still using reference back history
        sample['net_input']['attn_all_ref'] = [ x.detach().mean(dim=0, keepdim=True) for x in net_output_tf[1]['attn_all'] ]
        # if self.avg_attn:
        #     import pdb; pdb.set_trace()
        #     print(len(sample['net_input']['attn_all_ref']))
        #     print(sample['net_input']['attn_all_ref'][0].shape)
        
        # sample['net_input']['prev_output_tokens'][:,1:] = torch.topk(net_output_tf[0],1)[1].squeeze()[:,:-1].detach()
        net_output = model(**sample["net_input"])

        loss, nll_loss, att_loss = self.compute_loss_af(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "att_loss": att_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if 'attn_all_ref' in sample['net_input']: del sample['net_input']['attn_all_ref']
        return loss, sample_size, logging_output

    def compute_loss_af(self, model, net_output, sample, reduce=True):
        # import pdb; pdb.set_trace()
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        tmp, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        att_loss = l2_att_loss(
            [ a[:1] for a in net_output[1]['attn_all'] ], 
            sample['net_input']['attn_all_ref'],
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduce=reduce,
        ) * self.scale_attn_loss
        loss = tmp + att_loss
        return loss, nll_loss, att_loss