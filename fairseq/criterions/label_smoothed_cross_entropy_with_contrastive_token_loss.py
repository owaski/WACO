# Copyright (c) ByteDance, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ast import parse
import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

def padding_mask_to_lengths(padding_mask):
    return padding_mask.size(1) - torch.sum(padding_mask, dim=1)

@register_criterion("label_smoothed_cross_entropy_with_constrastive_token")
class LabelSmoothedCrossEntropyWithContrastiveTokenCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        contrastive_level='token',
        pooling='mean',
        cross_attention_num=10,
        ablation_type='ctc_cnn',
        ablation_weight=0.,
        use_dual_ctr=False,
        ctr_dropout_rate=0.0,
        enable_profiler=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        self.ablation_type = ablation_type
        self.ablation_weight = ablation_weight

        self.use_dual_ctr = use_dual_ctr
        self.ctr_dropout_rate = ctr_dropout_rate

        self.contrastive_level = contrastive_level
        self.cross_attention_num = cross_attention_num
        self.pooling = pooling

        self.enable_profiler = enable_profiler


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--contrastive-seqlen-type', default='src_text', type=str,
                            choices=['src_text', 'transcript',
                                     'audio_short', 'none'],
                            help='which type of length to times to the contrastive loss')
        parser.add_argument('--contrastive-level', type=str, default="",
                            help='contrastive level, token/sentence/cross_attention')
        parser.add_argument('--pooling', type=str, default='mean')

        parser.add_argument('--cross-attention-num', type=int, default=10)

        parser.add_argument('--ablation-type', type=str, default="ctc_cnn")
        parser.add_argument('--ablation-weight', type=float, default=0.)

        parser.add_argument("--use-dual-ctr", action="store_true",
                            help="if we want to use dual contrastive loss")
        parser.add_argument("--ctr-dropout-rate", default=0., type=float,
                            help='the dropout rate of hidden units')

        
        parser.add_argument('--enable-profiler', action='store_true', default=False)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        _net_output = model(**sample["net_input"]) # (x, extra)
        if model.training:
            net_output, encoder_out = _net_output
            contrastive_loss, short_audio_len = self.compute_contrastive_loss(
                model, sample, encoder_out,
                reduce=reduce, return_short_audio_len=True
            )
        else:
            net_output = _net_output
            contrastive_loss, short_audio_len = torch.tensor(0.0), None
        label_smoothed_nll_loss, nll_loss = torch.tensor(0.0), torch.tensor(0.0)
        if sample["target"] is not None: # ST triple dataset
            label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["target_ntokens"]
        )
        source_ntokens = sample["source_ntokens"]
        if label_smoothed_nll_loss is not None:
            loss = label_smoothed_nll_loss + self.contrastive_weight * contrastive_loss
        else:
            loss = contrastive_loss

        logging_output = {
            "loss": loss.data,
            "label_smoothed_nll_loss": label_smoothed_nll_loss.data,
            "nll_loss": nll_loss.data,
            "contrastive_loss": contrastive_loss.data,
            "source_ntokens": source_ntokens,
            "target_ntokens": sample["target_ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if nll_loss != 0:
            logging_output["ntokens"] = sample["target_ntokens"]

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_sequence_hidden(self, model, sample, packed_encoder_out, layer):
        if layer == 'emb':
            encoder_out = packed_encoder_out.encoder_embedding
        elif layer == 'last':
            encoder_out = packed_encoder_out.encoder_out
        else:
            raise NotImplementedError

        encoder_padding_mask = packed_encoder_out.encoder_padding_mask

        encoder_out = encoder_out.transpose(0, 1) # T x B x hid -> B x T x hid
        encoder_padding_mask = (~encoder_padding_mask).float()
        seq_hidden = (encoder_out * encoder_padding_mask.unsqueeze(-1)).sum(dim=1) / encoder_padding_mask.sum(dim=1).unsqueeze(-1)
        return seq_hidden

    def compute_contrastive_loss(self, model, sample,
                                 temp, layer='emb', reduce=True):
        
        loss = torch.tensor(0.)
        sample_size = 0

        static_model = getattr(self.task, "static_model", None)

        if sample["align_indices"].size(0) > 0:


            if self.contrastive_level == 'sentence':

                def _obtain_sent_feature():
                    s_encoder_out = model.encoder(
                    sample["align_input"]["src_tokens"], sample["align_input"]["src_lengths"]
                    )
                    with torch.no_grad():
                        t_encoder_out = model.encoder(
                            sample["align_input"]["src_text"], sample["align_input"]["src_text_lengths"]
                        )
                    return s_encoder_out, t_encoder_out
                
                def _compute_sent_loss(s_encoder_out, t_encoder_out):
                    audio_seq_hidden = self.get_sequence_hidden(model, sample, s_encoder_out, layer) # B x h
                    text_seq_hidden = self.get_sequence_hidden(model, sample, t_encoder_out, layer) # B x h
                    batch_size, hidden_size = audio_seq_hidden.size()
                    logits = F.cosine_similarity(audio_seq_hidden.expand((batch_size, batch_size, hidden_size)),
                                                text_seq_hidden.expand((batch_size, batch_size, hidden_size)).transpose(0, 1),
                                                dim=-1)
                    logits /= temp

                    if self.use_dual_ctr:
                        loss_audio = -torch.nn.LogSoftmax(0)(logits).diag()
                        loss_text = -torch.nn.LogSoftmax(1)(logits).diag()
                        loss = loss_audio + loss_text
                    else:
                        loss = -torch.nn.LogSoftmax(0)(logits).diag()

                    sample_size = batch_size

                    return loss, sample_size

                if self.enable_profiler:
                    import time
                    start_time = time.time()
                    s_encoder_out, t_encoder_out = _obtain_sent_feature()
                    torch.cuda.synchronize()
                    mid_time = time.time()
                    loss, sample_size = _compute_sent_loss(s_encoder_out, t_encoder_out)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    print('feature: {:.2f}ms, loss: {:.2f}ms, loss_ratio: {:.2f}'.format(
                        (mid_time - start_time) * 1000, (end_time - mid_time) * 1000, (end_time - mid_time) / (end_time - start_time)))
                else:
                    s_encoder_out, t_encoder_out = _obtain_sent_feature()
                    loss, sample_size = _compute_sent_loss(s_encoder_out, t_encoder_out)    
                
            elif self.contrastive_level == 'cross_attention':

                s_encoder_out = model.encoder(
                    sample["align_input"]["src_tokens"], sample["align_input"]["src_lengths"]
                )
                t_encoder_out = model.encoder(
                    sample["align_input"]["src_text"], sample["align_input"]["src_text_lengths"]
                )
                with torch.no_grad():
                    static_t_encoder_out = static_model.encoder(
                        sample["align_input"]["src_text"], sample["align_input"]["src_text_lengths"]
                    )

                s_x = s_encoder_out.encoder_out
                t_x = t_encoder_out.encoder_out
                static_t_x = static_t_encoder_out.encoder_out

                _, batch_size, hidden_size = s_x.size()

                query = torch.rand(self.cross_attention_num, 1, hidden_size).expand(
                    self.cross_attention_num, batch_size, hidden_size
                ).to(s_x)

                n_layer = len(model.decoder.layers)

                s_outputs = []
                static_t_outputs = []
                for l in range(n_layer):
                    
                    s_output, _ = static_model.decoder.layers[l].encoder_attn(
                        query=query,
                        key=s_x,
                        value=s_x,
                        key_padding_mask=s_encoder_out.encoder_padding_mask,
                        incremental_state=None,
                        static_kv=True,
                        need_weights=False,
                        need_head_weights=False,
                    )

                    static_t_output, _ = static_model.decoder.layers[l].encoder_attn(
                        query=query,
                        key=static_t_x,
                        value=static_t_x,
                        key_padding_mask=t_encoder_out.encoder_padding_mask,
                        incremental_state=None,
                        static_kv=True,
                        need_weights=False,
                        need_head_weights=False,
                    )

                    s_outputs.append(s_output)
                    static_t_outputs.append(static_t_output)

                s_outputs = torch.cat(s_outputs, dim=0)
                static_t_outputs = torch.cat(static_t_outputs, dim=0)
                
                logits = F.cosine_similarity(
                    s_outputs.unsqueeze(2),
                    static_t_outputs.unsqueeze(1),
                    dim=-1
                )
                logits /= temp

                loss = -torch.nn.LogSoftmax(2)(logits).mean(dim=0).diag().sum()

                reg_loss = (t_x - static_t_x).norm(dim=-1).sum() * 0.01

                # print(loss.item() / batch_size, reg_loss.item() / batch_size)

                loss = loss + reg_loss

                sample_size = batch_size

            elif self.contrastive_level == 'cross_attention_no_reg':

                s_encoder_out = model.encoder(
                    sample["align_input"]["src_tokens"], sample["align_input"]["src_lengths"]
                )
                
                with torch.no_grad():
                    t_encoder_out = model.encoder(
                        sample["align_input"]["src_text"], sample["align_input"]["src_text_lengths"]
                    )

                s_x = s_encoder_out.encoder_out
                t_x = t_encoder_out.encoder_out

                _, batch_size, hidden_size = s_x.size()

                query = torch.rand(self.cross_attention_num, 1, hidden_size).expand(
                    self.cross_attention_num, batch_size, hidden_size
                ).to(s_x)

                n_layer = len(model.decoder.layers)

                s_outputs = []
                t_outputs = []
                for l in range(n_layer):
                    
                    s_output, _ = model.decoder.layers[l].encoder_attn(
                        query=query,
                        key=s_x,
                        value=s_x,
                        key_padding_mask=s_encoder_out.encoder_padding_mask,
                        incremental_state=None,
                        static_kv=True,
                        need_weights=False,
                        need_head_weights=False,
                    )

                    t_output, _ = model.decoder.layers[l].encoder_attn(
                        query=query,
                        key=t_x,
                        value=t_x,
                        key_padding_mask=t_encoder_out.encoder_padding_mask,
                        incremental_state=None,
                        static_kv=True,
                        need_weights=False,
                        need_head_weights=False,
                    )

                    s_outputs.append(s_output)
                    t_outputs.append(t_output)

                s_outputs = torch.cat(s_outputs, dim=0)
                t_outputs = torch.cat(t_outputs, dim=0)
                
                logits = F.cosine_similarity(
                    s_outputs.unsqueeze(2),
                    t_outputs.unsqueeze(1),
                    dim=-1
                )
                logits /= temp

                loss = -torch.nn.LogSoftmax(2)(logits).mean(dim=0).diag().sum()

                sample_size = batch_size

            elif self.contrastive_level == 'cross_attention_cos':

                s_encoder_out = model.encoder(
                    sample["align_input"]["src_tokens"], sample["align_input"]["src_lengths"]
                )
                with torch.no_grad():
                    t_encoder_out = model.encoder(
                        sample["align_input"]["src_text"], sample["align_input"]["src_text_lengths"]
                    )

                s_x = s_encoder_out.encoder_out
                t_x = t_encoder_out.encoder_out

                _, batch_size, hidden_size = s_x.size()

                query = torch.rand(self.cross_attention_num, 1, hidden_size).expand(
                    self.cross_attention_num, batch_size, hidden_size
                ).to(s_x)

                n_layer = len(model.decoder.layers)

                s_outputs = []
                t_outputs = []
                for l in range(n_layer):
                    for param in model.decoder.layers[l].encoder_attn.parameters():
                        param.requires_grad = False
                    
                    s_output, _ = model.decoder.layers[l].encoder_attn(
                        query=query,
                        key=s_x,
                        value=s_x,
                        key_padding_mask=s_encoder_out.encoder_padding_mask,
                        incremental_state=None,
                        static_kv=True,
                        need_weights=False,
                        need_head_weights=False,
                    )

                    t_output, _ = model.decoder.layers[l].encoder_attn(
                        query=query,
                        key=t_x,
                        value=t_x,
                        key_padding_mask=t_encoder_out.encoder_padding_mask,
                        incremental_state=None,
                        static_kv=True,
                        need_weights=False,
                        need_head_weights=False,
                    )

                    s_outputs.append(s_output)
                    t_outputs.append(t_output)

                s_outputs = torch.cat(s_outputs, dim=0)
                t_outputs = torch.cat(t_outputs, dim=0)

                loss = F.cosine_similarity(
                    s_outputs,
                    t_outputs,
                    dim=-1
                ).mean(dim=0)

                sample_size = batch_size

            elif self.contrastive_level == 'cross_attention_l2':

                s_encoder_out = model.encoder(
                    sample["align_input"]["src_tokens"], sample["align_input"]["src_lengths"]
                )
                with torch.no_grad():
                    t_encoder_out = model.encoder(
                        sample["align_input"]["src_text"], sample["align_input"]["src_text_lengths"]
                    )

                s_x = s_encoder_out.encoder_out
                t_x = t_encoder_out.encoder_out

                _, batch_size, hidden_size = s_x.size()

                query = torch.rand(self.cross_attention_num, 1, hidden_size).expand(
                    self.cross_attention_num, batch_size, hidden_size
                ).to(s_x)

                n_layer = len(model.decoder.layers)

                s_outputs = []
                t_outputs = []
                for l in range(n_layer):
                    for param in model.decoder.layers[l].encoder_attn.parameters():
                        param.requires_grad = False
                    
                    s_output, _ = model.decoder.layers[l].encoder_attn(
                        query=query,
                        key=s_x,
                        value=s_x,
                        key_padding_mask=s_encoder_out.encoder_padding_mask,
                        incremental_state=None,
                        static_kv=True,
                        need_weights=False,
                        need_head_weights=False,
                    )

                    t_output, _ = model.decoder.layers[l].encoder_attn(
                        query=query,
                        key=t_x,
                        value=t_x,
                        key_padding_mask=t_encoder_out.encoder_padding_mask,
                        incremental_state=None,
                        static_kv=True,
                        need_weights=False,
                        need_head_weights=False,
                    )

                    s_outputs.append(s_output)
                    t_outputs.append(t_output)

                s_outputs = torch.cat(s_outputs, dim=0)
                t_outputs = torch.cat(t_outputs, dim=0)

                loss = (s_outputs - t_outputs).norm(dim=-1).mean(dim=0)

                sample_size = batch_size

            elif self.contrastive_level == 'token':

                def _obtain_feature():
                    s_encoder_out = model.encoder(
                        sample["align_input"]["src_tokens"], sample["align_input"]["src_lengths"]
                    )
                    with torch.no_grad():
                        t_encoder_out = model.encoder(
                            sample["align_input"]["src_text"], sample["align_input"]["src_text_lengths"]
                        )
                    return s_encoder_out, t_encoder_out

                def _compute_loss(s_encoder_out, t_encoder_out):

                    if layer == 'emb':
                        s_x = s_encoder_out.encoder_embedding
                        t_x = t_encoder_out.encoder_embedding
                    elif layer == 'last':
                        s_x = s_encoder_out.encoder_out
                        t_x = t_encoder_out.encoder_out
                    else:
                        raise NotImplementedError
                        
                    s_padding_mask = s_encoder_out.encoder_padding_mask
                    t_padding_mask = t_encoder_out.encoder_padding_mask

                    s_len = padding_mask_to_lengths(s_padding_mask)
                    t_len = padding_mask_to_lengths(t_padding_mask)

                    bsz = s_x.size(1)

                    loss = torch.tensor(0.)

                    s_f = []
                    t_f = []

                    align = sample["align"]

                    s_x = s_x.float()
                    t_x = t_x.float()

                    for i in range(bsz):
                        segment, interval = align[i]
                        
                        for (t_l, t_r), (s_l, s_r) in zip(segment, interval):
                            s_l = int((s_l * s_len[i]).floor())
                            s_r = int((s_r * s_len[i]).ceil())

                            if self.pooling == 'mean':
                                t_feature = t_x[t_l : t_r + 1, i].mean(dim=0)
                                s_feature = s_x[s_l : s_r + 1, i].mean(dim=0)
                            elif self.pooling == 'max':
                                t_feature = t_x[t_l : t_r + 1, i].max(dim=0)[0]
                                s_feature = s_x[s_l : s_r + 1, i].max(dim=0)[0]
                            elif self.pooling == 'sum':
                                t_feature = t_x[t_l : t_r + 1, i].sum(dim=0)
                                s_feature = s_x[s_l : s_r + 1, i].sum(dim=0)
                            else:
                                raise NotImplementedError

                            s_f.append(s_feature)
                            t_f.append(t_feature)

                    s_f = torch.stack(s_f, dim=0)
                    t_f = torch.stack(t_f, dim=0)

                    logits = F.cosine_similarity(
                        s_f.unsqueeze(1),
                        t_f.unsqueeze(0),
                        dim=-1
                    ) / temp

                    label = torch.arange(s_f.size(0)).to(logits.device)

                    loss = F.cross_entropy(logits, label, reduction='sum')
                    sample_size = s_f.size(0)

                    return loss, sample_size

                if self.enable_profiler:
                    import time
                    start_time = time.time()
                    s_encoder_out, t_encoder_out = _obtain_feature()
                    torch.cuda.synchronize()
                    mid_time = time.time()
                    loss, sample_size = _compute_loss(s_encoder_out, t_encoder_out)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    print('feature: {:.2f}ms, loss: {:.2f}ms, loss_ratio: {:.2f}'.format(
                        (mid_time - start_time) * 1000, (end_time - mid_time) * 1000, (end_time - mid_time) / (end_time - start_time)))
                else:
                    s_encoder_out, t_encoder_out = _obtain_feature()
                    loss, sample_size = _compute_loss(s_encoder_out, t_encoder_out)

            elif self.ablation_weight > 0:

                ctc_logit, padding_mask, lprobs = model.encoder.compute_ctc_logit_and_logprob(
                    sample["align_input"]["src_tokens"], sample["align_input"]["src_lengths"])
                lprobs = lprobs.contiguous()
                non_padding_mask = ~padding_mask
                input_lengths = non_padding_mask.long().sum(-1)

                if self.ablation_type == 'ctc_phoneme':
                    phonemes = sample["align_input"]["src_phoneme"]
                    n_phonemes = sample["align_input"]["src_phoneme_lengths"]
                    transcript_flat = []
                    for i, n_phoneme in enumerate(n_phonemes):
                        transcript_flat.append(phonemes[i, :n_phoneme])
                    transcript_flat = torch.cat(transcript_flat, dim=0)
                    transcript_lengths = n_phonemes
                else:
                    transcript = sample["align_input"]["src_text"]
                    if self.ignore_prefix_size > 0:
                        transcript = transcript[:, self.ignore_prefix_size:]
                    pad_mask = (transcript != self.task.tgt_dict.pad()) & (transcript != self.task.tgt_dict.eos())
                    transcript_flat = transcript.masked_select(pad_mask)
                    transcript_lengths = pad_mask.sum(-1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss = F.ctc_loss(
                        lprobs,
                        transcript_flat,
                        input_lengths,
                        transcript_lengths,
                        blank=len(self.task.src_dict) - 1 if self.ablation_type == 'ctc_phoneme' else self.task.tgt_dict.bos(),
                        reduction="sum" if reduce else "none",
                        zero_infinity=True,
                    )

                sample_size = transcript_lengths.sum()


        if reduce:
            loss = loss.sum()
        return loss, sample_size

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        label_smoothed_nll_loss_sum = sum(log.get("label_smoothed_nll_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        target_ntokens = sum(log.get("target_ntokens", 0) for log in logging_outputs)
        source_ntokens = sum(log.get("source_ntokens", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "label_smoothed_nll_loss", label_smoothed_nll_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / target_ntokens / math.log(2), target_ntokens, round=3
        )
        metrics.log_scalar(
            "contrasitve_loss", contrastive_loss_sum / nsentences / math.log(2), nsentences, round=3
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