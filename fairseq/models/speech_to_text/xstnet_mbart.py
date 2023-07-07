#!/usr/bin/env python3

from argparse import Namespace
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import copy
import numpy as np

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils, tasks
from fairseq.data.data_utils import lengths_to_padding_mask, compute_mask_indices
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.model_parallel.models.transformer import TransformerEncoder
from fairseq.models.transformer import Embedding, Linear
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    LayerDropModuleList
)
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler, TransformerDecoder
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2VecCtc
from fairseq.models.speech_to_text.s2t_w2v2_transformer import S2TTransformerModelW2V2
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


from torch import Tensor


logger = logging.getLogger(__name__)


@register_model("xstnet_mbart")
class XSTNetMBART(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.is_text_input = False # default

    @staticmethod
    def add_args(parser):
        S2TTransformerModelW2V2.add_args(parser)
        parser.add_argument("--textual-encoder-embed-dim", type=int, metavar="N",
                            help="encoder embded dim for text input")
        parser.add_argument("--no-subsample-audio", action='store_true')
        parser.add_argument("--no-audio-pretrain", action='store_true')
        # mBART50 dir
        parser.add_argument(
            '--mbart50-dir', 
            type=str, 
            metavar='STR',
            help='directory to mbart50 model'
        )

    @classmethod
    def build_encoder(cls, args, dict, embed_tokens):
        encoder = XSTNetMBARTEncoder(args, dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, dict, embed_tokens):
        return TransformerDecoder(args, dict, embed_tokens)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        xstnet_mbart_base_architecture(args)

        print(args)

        encoder_embedding = None
        task.src_dict = getattr(task, "src_dict", task.tgt_dict)
        encoder_embedding = cls.build_embedding(args, task.target_dictionary, args.encoder_embed_dim)
        decoder_embedding = cls.build_embedding(args, task.target_dictionary, args.decoder_embed_dim)
        encoder = cls.build_encoder(args, task.source_dictionary, encoder_embedding)
        decoder = cls.build_decoder(args, task.target_dictionary, decoder_embedding)

        mbart50_params = checkpoint_utils.load_checkpoint_to_cpu(os.path.join(args.mbart50_dir, 'model.pt'))['model']
        mbart50_encoder_params = {k[8:]: v for k, v in mbart50_params.items() if k.startswith('encoder.')}
        mbart50_decoder_params = {k[8:]: v for k, v in mbart50_params.items() if k.startswith('decoder.')}
        encoder.transformer_encoder.load_state_dict(mbart50_encoder_params)
        decoder.load_state_dict(mbart50_decoder_params, strict=False)

        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def set_mt_only(self):
        self.is_text_input = True
        self.encoder.is_text_input = True

    def forward(self, src_tokens, src_lengths, prev_output_tokens,
                is_text_input=False, **kwargs):
        if self.is_text_input:
            is_text_input = True
        encoder_out = self.encoder(src_tokens, src_lengths, is_text_input=is_text_input)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens,
                                   encoder_out=encoder_out)
        if self.training:
            return decoder_out, encoder_out
        return decoder_out


class XSTNetMBARTEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.args = args
        self.embed_tokens = embed_tokens
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.padding_idx = embed_tokens.padding_idx
        self.textual_encoder_embed_dim = embed_tokens.embedding_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.textual_encoder_embed_dim)

        self._build_acoustic_encoder(args)
        self._build_textual_encoder(args)
        self.is_text_input = False

        # CTC module
        self.use_ctc = (("ctc" in getattr(args, "ablation_type", "")) and (getattr(args, "ablation_weight", 0.0) > 0)) \
            or (("ctc" in getattr(args, "criterion", "")) and (getattr(args, "ctc_weight", 0.0) > 0))
        if self.use_ctc:
            if (getattr(args, "ablation_type", False) == "ctc_cnn") or \
                    (getattr(args, "ctc_type", False) == "ctc_cnn"):
                self.ctc_type = "ctc_cnn"
                self.ctc_projection = nn.Linear(
                    embed_tokens.embedding_dim,
                    embed_tokens.weight.shape[0],
                    bias=False,
                )
                self.ctc_projection.weight = embed_tokens.weight
            elif getattr(args, "ablation_type", False) == "ctc_phoneme":
                self.ctc_type = "ctc_phoneme"
                self.ctc_projection = nn.Linear(
                    self.w2v_args.encoder_embed_dim,
                    len(dictionary),
                    bias=False
                )
            elif (getattr(args, "ablation_type", False) == "ctc_w2v") or \
                    (getattr(args, "ctc_type", False) == "ctc_w2v"):
                self.ctc_type = "ctc_w2v"
                self.ctc_projection = nn.Linear(
                    self.w2v_args.encoder_embed_dim,
                    embed_tokens.weight.shape[0],
                )
            self.ctc_softmax = nn.Softmax(dim=-1)

    def _build_acoustic_encoder(self, args):
        assert args.w2v2_model_path is not None
        self.w2v2_model_path = args.w2v2_model_path
        self.use_asr_finetune_w2v = args.use_asr_finetune_w2v
        try:
            ckpt = torch.load(self.w2v2_model_path)
        except FileNotFoundError:
            if not os.path.exists("wav2vec_small.pt"):
                os.system(f"wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt")
            ckpt = torch.load("wav2vec_small.pt")
        self.w2v_args = ckpt["args"]
        if not self.use_asr_finetune_w2v:  # if use ssl-trained only
            self.w2v_args = ckpt["args"]
            self.wav2vec_model = Wav2Vec2Model.build_model(ckpt['args'], task=None)
            if not getattr(args, "no_audio_pretrain", False):
                self.wav2vec_model.load_state_dict(ckpt['model'])
        else:  # wav2vec-ctc model
            ckpt["args"].data = args.data
            if not os.path.exists(os.path.join(ckpt["args"].data, f"dict.{ckpt['args'].labels}.txt")):
                os.system(f"wget -P {ckpt['args'].data} https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt")
            task = tasks.setup_task(ckpt["args"])
            model_finetuned = Wav2VecCtc.build_model(ckpt["args"], task=task)
            model_finetuned.load_state_dict(ckpt['model'])
            self.wav2vec_model = model_finetuned.w2v_encoder.w2v_model
            self.w2v_args = ckpt["args"].w2v_args["model"]
        self.freeze_w2v = args.freeze_w2v

        w2v_output_dim = self.w2v_args.encoder_embed_dim
        self.no_subsample_audio = False
        self.subsample_audio = Conv1dSubsampler(
            w2v_output_dim,
            args.conv_channels,
            self.textual_encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

    def _build_textual_encoder(self, args, ):
        self.max_source_positions = args.max_source_positions

        t_args = copy.deepcopy(args)
        t_args.max_source_positions = min(t_args.max_source_positions, 1024)
        self.transformer_encoder = _TransformerEncoder(t_args, self.dictionary, self.embed_tokens)

        self.embed_scale = 1.0 if args.no_scale_embedding else np.sqrt(args.encoder_embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                self.textual_encoder_embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        # if getattr(args, "layernorm_embedding", False):
        #     self.layernorm_embedding = LayerNorm(self.textual_encoder_embed_dim)
        # else:
        #     self.layernorm_embedding = None
        # self.transformer_layers = nn.ModuleList(
        #     [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        # )
        # if args.encoder_normalize_before:
        #     self.layer_norm = LayerNorm(self.textual_encoder_embed_dim)
        # else:
        #     self.layer_norm = None

    def _get_w2v_feature(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(src_tokens, padding_mask)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return w2v_feature, padding_mask, output_length

    def embedding_audio(self, src_tokens, src_lengths):
        if self.freeze_w2v:
            with torch.no_grad():
                w2v_feature, w2v2_padding_mask, input_lengths = self._get_w2v_feature(
                    src_tokens, src_lengths)
        else:
            w2v_feature, w2v2_padding_mask, input_lengths = self._get_w2v_feature(
                src_tokens, src_lengths)

        x, input_lengths = self.subsample_audio(w2v_feature, input_lengths)
        x = x.transpose(0, 1)

        x = self.embed_scale * x
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask)
        x += positions
        x = self.dropout_module(x)

        return x, encoder_padding_mask, input_lengths

    def embedding_text(self, src_tokens, src_lengths):
        token_embedding = self.embed_tokens(src_tokens)
        x = self.embed_scale * token_embedding

        encoder_padding_mask = lengths_to_padding_mask(src_lengths)

        x += self.transformer_encoder.embed_positions(encoder_padding_mask)

        if self.transformer_encoder.layernorm_embedding is not None:
            x = self.transformer_encoder.layernorm_embedding(x)
        x = self.dropout_module(x)

        return x, encoder_padding_mask, src_lengths

    def forward(self, src_tokens, src_lengths, is_text_input=False, **kwargs):
        """
        src_tokens: b x seq, float tensor if it is audio input, LongTensor if it is text input
        src_lengths: b-dim LongTensor
        """
        if self.is_text_input or not src_tokens.dtype.is_floating_point:
            is_text_input = True
        if is_text_input:
            x, encoder_padding_mask, input_lengths = self.embedding_text(src_tokens, src_lengths)
        else:
            x, encoder_padding_mask, input_lengths = self.embedding_audio(src_tokens, src_lengths)
        encoder_embedding = x.transpose(0, 1)
        # 3. Transformer-layers
        # for layer in self.transformer_layers:
        #     x = layer(x, encoder_padding_mask)
        # if self.layer_norm is not None:
        #     x = self.layer_norm(x)
        x = self.transformer_encoder(x, input_lengths).encoder_out

        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=encoder_embedding,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
            output_encoder_lengths=input_lengths
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """

        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )

        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
            output_encoder_lengths=None
        )

    def compute_ctc_logit_and_logprob(self, src_tokens, src_lengths):
        assert self.use_ctc, "CTC is not available!"
        w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
            src_tokens, src_lengths
        )
        encoder_state = w2v_feature # b x seq x 768
        if self.ctc_type == "ctc_cnn":
            encoder_state, input_lengths = self.subsample_audio(w2v_feature, input_lengths) # seq x b x 512
            encoder_state = encoder_state.transpose(0, 1) # b x seq x 512
            encoder_state = self.embed_scale * encoder_state
            encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        elif self.ctc_type == "ctc_phoneme":
            encoder_state = w2v_feature
        else:
            assert self.ctc_type == "ctc_w2v", "ctc type should be ctc_w2v or ctc_cnn or ctc_phoneme"
        encoder_state = self.dropout_module(encoder_state)

        # ctc_logit = self.ctc_projection(encoder_state) # b x seq x voc
        if self.ctc_type == "ctc_cnn":
            ctc_logit = torch.matmul(encoder_state, self.embed_tokens.weight.T.detach())
        elif self.ctc_type == "ctc_phoneme":
            ctc_logit = self.ctc_projection(encoder_state)

        logits = ctc_logit.float()
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1) # seq x b x voc
        return ctc_logit, encoder_padding_mask, log_probs


class _TransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(TransformerEncoder, self).__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args, i) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args, idx):
        return _TransformerEncoderLayer(args, idx)

    def forward(
        self,
        token_embeddings,
        src_lengths,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = lengths_to_padding_mask(src_lengths)

        # x, encoder_embedding = self.forward_embedding(encoder_padding_mask, token_embeddings)

        # B x T x C -> T x B x C
        x = token_embeddings.transpose(0, 1)

        encoder_states = [] if return_all_hiddens else None
        
        # encoder layers
        for layer in self.layers:
            x, internal = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(internal)

        if self.layer_norm is not None:
            x = self.layer_norm(x)      

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=token_embeddings,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
            output_encoder_lengths=None,
        )


class _TransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, idx):
        super().__init__(args)
        self.remove_residual = False

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)


        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        internal_state = x
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        if not self.remove_residual: # optional to remove residual connection
            x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, internal_state


@register_model_architecture(model_name="xstnet_mbart", arch_name="xstnet_mbart_base")
def xstnet_mbart_base_architecture(args):
    # wav2vec2
    args.w2v2_model_path = getattr(args, "w2v2_model_path", "./wav2vec_small_100h.pt")
    args.freeze_w2v = getattr(args, "freeze_w2v", False) # default is false, 'store_true'
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)

     # Convolutional subsampler
    args.cnn_subsampler = getattr(args, 'cnn_subsampler', True)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)

    # mBART-large config
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", True)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0.0)