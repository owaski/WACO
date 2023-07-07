# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from yaml import parse
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.criterions.label_smoothed_cross_entropy_with_contrastive_token_loss import \
    LabelSmoothedCrossEntropyWithContrastiveTokenCriterion
from fairseq.data.data_utils import lengths_to_padding_mask


@register_criterion("multi_task_cross_entropy_with_contrastive_token_ssl_with_extra_MT")
class MultiTaskCrossEntropyWithContrastiveTokenSSLWithExtraMT(LabelSmoothedCrossEntropyWithContrastiveTokenCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        no_asr=False,
        contrastive_weight=[0., 0.],
        contrastive_temperature=[1.0, 1.0],
        contrastive_level='token',
        cross_attention_num=10,
        ablation_type=None,
        ablation_weight=0.,
        enable_profiler=False,
        ssl_weight=0.,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy,
                         contrastive_level, cross_attention_num, ablation_type, ablation_weight, enable_profiler=enable_profiler)
        assert len(contrastive_weight) == len(contrastive_temperature)
        self.no_asr = no_asr
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.ssl_weight = ssl_weight

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyWithContrastiveTokenCriterion.add_args(parser)
        parser.add_argument('--no-asr', action='store_true')
        parser.add_argument('--contrastive-weight', default=[0., 0.], type=float, nargs='+',
                            help='the weight of contrastive loss')
        parser.add_argument('--contrastive-temperature', default=[1.0, 1.0], type=float, nargs='+',
                            help='the temperature in the contrastive loss')
        parser.add_argument('--ssl-weight', default=0., type=float, 
                            help='the weight of ssl loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        label_smoothed_nll_loss, nll_loss = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss_asr, nll_loss_asr = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss_mt, nll_loss_mt = torch.tensor(0.0), torch.tensor(0.0)
        contrastive_loss, short_audio_len = torch.tensor(0.0), None
        contrastive_semantic_loss = torch.tensor(0.0)

        if "mode" in sample["net_input"] and sample["net_input"]["mode"] == "text_to_text":
            sample["dataset_type"] = "mt"
            sample["net_input"]["is_text_input"] = True
        else:
            sample["net_input"]["is_text_input"] = False

        # _net_output = model(**sample["net_input"])  # (x, extra)
        # if model.training:
        #     net_output, encoder_out = _net_output
        #     if (sample["dataset_type"] != "mt") and (self.contrastive_weight > 0):
        #         contrastive_loss, short_audio_len = self.compute_contrastive_loss(
        #             model, sample, encoder_out,
        #             reduce=reduce, return_short_audio_len=True
        #         )
        # else:
        #     net_output = _net_output

        asr_sample_size = 0
        contrastive_sample_size = 0
        ssl_sample_size = 0
        if sample["dataset_type"] == "st":
            if sample["target"] is not None:
                label_smoothed_nll_loss, nll_loss, net_output = self.compute_loss_st(model, sample, reduce=reduce)
                if not self.no_asr:
                    label_smoothed_nll_loss_asr, nll_loss_asr = self.compute_loss_asr(model, sample, reduce=reduce)
                label_smoothed_nll_loss_mt, nll_loss_mt = self.compute_loss_mt(model, sample, reduce=reduce)
            if self.ablation_weight > 0:
                contrastive_loss, contrastive_sample_size = self.compute_contrastive_loss(model, sample, 0, layer='emb', reduce=reduce)
            else:
                if len(self.contrastive_weight) > 0 and self.contrastive_weight[0] > 0:
                    contrastive_loss, contrastive_sample_size = self.compute_contrastive_loss(model, sample, self.contrastive_temperature[0], layer='emb', reduce=reduce)
                if len(self.contrastive_weight) > 1 and self.contrastive_weight[1] > 0:
                    contrastive_semantic_loss, _ = self.compute_contrastive_loss(model, sample, self.contrastive_temperature[1], layer='last', reduce=reduce)

            if self.ssl_weight > 0:
                ssl_loss, ssl_sample_size = self.compute_ssl_loss(model, sample, reduce=reduce)

        else:  # mt type compute CE_mt loss
            _net_output = model(**sample["net_input"])  # (x, extra)
            if model.training:
                net_output, encoder_out = _net_output
            else:
                net_output = _net_output
            label_smoothed_nll_loss_mt, nll_loss_mt = self.compute_loss(model, net_output, sample, reduce=reduce)

        if sample["dataset_type"] == "st":
            source_ntokens = sample["source_ntokens"]
            target_ntokens = sample["target_ntokens"]
            target_ntokens_st = target_ntokens
            target_ntokens_mt = 0
            sample_size = sample["source"].size(0) if self.sentence_avg else sample["ntokens"]
            asr_sample_size = sample["mt_src"]["src_tokens"].size(0) if self.sentence_avg else sample["source_lengths"][sample["st_indices"]].sum()
        else:
            source_ntokens = 0
            target_ntokens = sample["ntokens"]
            target_ntokens_mt = target_ntokens
            target_ntokens_st = 0
            sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]

        nsentences = sample["source"].size(0)
        if sample["dataset_type"] == "st":
            multi_ce_loss = label_smoothed_nll_loss + label_smoothed_nll_loss_asr + label_smoothed_nll_loss_mt
            if self.ablation_weight == 0:
                loss = multi_ce_loss
                if len(self.contrastive_weight) > 0 and self.contrastive_weight[0] > 0:
                    loss = loss + self.contrastive_weight[0] * contrastive_loss
                if len(self.contrastive_weight) > 1 and self.contrastive_weight[1] > 0:
                    loss = loss + self.contrastive_weight[1] * contrastive_semantic_loss
            else:
                loss = multi_ce_loss + self.ablation_weight * contrastive_loss

            if self.ssl_weight > 0:
                loss = loss + self.ssl_weight * ssl_loss
        else:
            loss = label_smoothed_nll_loss_mt

        # print(ssl_loss)

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "contrastive_loss": contrastive_loss.data,
            "contrastive_semantic_loss": contrastive_semantic_loss.data,
            "ssl_loss": ssl_loss.data.float(),
            "source_ntokens": source_ntokens,
            "target_ntokens": target_ntokens,
            "target_ntokens_mt": target_ntokens_mt,
            "target_ntokens_st": target_ntokens_st,
            "ntokens": target_ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "asr_sample_size": asr_sample_size,
            "contrastive_sample_size": contrastive_sample_size,
            "ssl_sample_size": ssl_sample_size,
            "nll_loss_asr": nll_loss_asr.data,
            "nll_loss_mt": nll_loss_mt.data,
            "st_nsentences": nsentences if sample["dataset_type"] != "mt" else 0,
            "mt_nsentences": nsentences if sample["dataset_type"] == "mt" else 0,
        }

        # print(sample["st_indices"], sample["target"])

        if self.report_accuracy and sample["st_indices"].size(0) > 0:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_ssl_loss(self, model, sample, reduce=True):
        src_tokens = sample["align_input"]["src_tokens"]
        src_lengths = sample["align_input"]["src_lengths"]
        padding_mask = lengths_to_padding_mask(src_lengths)
        output = model.encoder.wav2vec_model(src_tokens, padding_mask)
        logits = output['x']     
        logits = logits.view(logits.size(0), -1).T
        labels = torch.zeros(logits.size(0), dtype=int).to(logits.device)
        loss = F.cross_entropy(logits, labels, reduction='sum' if reduce else 'none')
        return loss, labels.size(0)

    def compute_loss_st(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        if model.training:
            net_output = net_output[0]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample["target"]
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, net_output

    def compute_loss_asr(self, model, sample, reduce=True):
        net_output = model(**sample["asr_input"])
        if model.training:
            net_output = net_output[0]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample["mt_input"]["src_tokens"]
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_loss_mt(self, model, sample, reduce=True):
        net_output = model(**sample["mt_input"], is_text_input=True)
        if model.training:
            net_output = net_output[0]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample["target"]
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        contrastive_semantic_loss_sum = sum(log.get("contrastive_semantic_loss", 0) for log in logging_outputs)
        ssl_loss_sum = sum(log.get("ssl_loss", 0) for log in logging_outputs)
        target_ntokens = sum(log.get("target_ntokens", 0) for log in logging_outputs)
        source_ntokens = sum(log.get("source_ntokens", 0) for log in logging_outputs)
        target_ntokens_mt = sum(log.get("target_ntokens_mt", 0) for log in logging_outputs)
        target_ntokens_st = sum(log.get("target_ntokens_st", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        mt_nsentences = sum(log.get("mt_nsentences", 0) for log in logging_outputs)
        st_nsentences = sum(log.get("st_nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        asr_sample_size = sum(log.get("asr_sample_size", 0) for log in logging_outputs)
        contrastive_sample_size = sum(log.get("contrastive_sample_size", 0) for log in logging_outputs)
        ssl_sample_size = sum(log.get("ssl_sample_size", 0) for log in logging_outputs)

        nll_loss_sum_asr = sum(log.get("nll_loss_asr", 0) for log in logging_outputs)
        nll_loss_sum_mt = sum(log.get("nll_loss_mt", 0) for log in logging_outputs)

        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), sample_size, round=3)
        if target_ntokens_st > 0:
            metrics.log_scalar("nll_loss",
                            nll_loss_sum / target_ntokens_st / math.log(2), target_ntokens_st, round=3)
            metrics.log_scalar("nll_loss_asr",
                            nll_loss_sum_asr / asr_sample_size / math.log(2), asr_sample_size, round=3)
            metrics.log_scalar("nll_loss_mt",
                            nll_loss_sum_mt / target_ntokens / math.log(2), target_ntokens, round=3)
        if contrastive_sample_size > 0:
            metrics.log_scalar("contrastive_loss",
                            contrastive_loss_sum / contrastive_sample_size / math.log(2), contrastive_sample_size, round=3)
            metrics.log_scalar("contrastive_semantic_loss",
                            contrastive_semantic_loss_sum / contrastive_sample_size / math.log(2), contrastive_sample_size, round=3)
        
        if ssl_sample_size > 0:
            print(ssl_loss_sum, ssl_sample_size)
            metrics.log_scalar("ssl_loss",
                            ssl_loss_sum / ssl_sample_size / math.log(2), ssl_sample_size, round=3)

        metrics.log_scalar("bsz_st", st_nsentences, priority=190, round=1)
        metrics.log_scalar("bsz_mt", mt_nsentences, priority=190, round=1)

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