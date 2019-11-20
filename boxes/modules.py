import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
from .box_wrapper import SigmoidBoxTensor, BoxTensor, TBoxTensor, DeltaBoxTensor
from typing import (List, Tuple, Dict, Optional, Any, Union, TypeVar, Type,
                    Callable)
from allennlp.modules.seq2vec_encoders import pytorch_seq2vec_wrapper
from allennlp.modules.token_embedders import Embedding
import logging
logger = logging.getLogger(__name__)

TTensor = TypeVar("TTensor", bound="torch.Tensor")

#TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")



@torch.no_grad()
def mask_from_lens(lens: List[int],
                   t: Optional[torch.Tensor] = None,
                   value: Union[int, float, torch.Tensor] = 1.):

    if t is None:
        t = torch.zeros(len(lens), max(lens))

    if t.size(0) != len(lens):
        raise ValueError(
            "t.size(0) should be equal to len(lens) but are {} and {}".format(
                t.size(0), len(lens)))

    for i, l in enumerate(lens):
        t[i][list(range(l))] = value

    return t


class LSTMBox(torch.nn.LSTM):
    """Module with standard lstm at the bottom but Boxes at the output"""

    def __init__(self, *args, box_type='SigmoidBoxes', **kwargs):
        # make sure that number of hidden dim is even
        # hidden_dim = args[1] * 2 if kwargs.get(
        # 'bidirectional', default=False) else args[1]
        self.box_type = box_type
        hidden_dim = args[1]

        if hidden_dim % 2 != 0:
            raise ValueError(
                "hidden_dim  has to be even but is {}".format(hidden_dim))
        super().__init__(*args, **kwargs)
        self.boxes = BoxView(box_type, split_dim=-1)

    def forward(self,
                inp: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[TBoxTensor, Tuple[Tensor, Tensor]]:
        # get lstm's output
        output, (h_n, c_n) = super().forward(inp, hx=hx)
        packed_inp = False

        # check if packed. If so, unpack

        if isinstance(output, torch.nn.utils.rnn.PackedSequence):
            packed_inp = True
            output, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=self.batch_first)
        # when LSTM is bidirectional, the output of both directions is used in
        # z as well as Z. Hence, output of both directions is split

        if self.bidirectional:

            if self.batch_first:
                seq_len = output.size(1)
                batch = output.size(0)
                split_on_dir = output.view(batch, seq_len, 2, self.hidden_size)
                output_dir1 = split_on_dir[..., 0, :]
                output_dir2 = split_on_dir[..., 1, :]
            else:  # self.batch_first == False:
                seq_len = output.size(0)
                batch = output.size(1)
                split_on_dir = output.view(seq_len, batch, 2, self.hidden_size)
                output_dir1 = split_on_dir[..., 0, :]
                output_dir2 = split_on_dir[..., 1, :]

            boxes_dir1 = self.boxes(output_dir1)
            boxes_dir2 = self.boxes(output_dir2)
            box_output = self.boxes.box_types[self.box_type].cat((boxes_dir1,
                                                                  boxes_dir2))

        else:
            box_output = self.boxes(output)

        return box_output.data, (h_n, c_n)


class PytorchSeq2BoxWrapper(pytorch_seq2vec_wrapper.PytorchSeq2VecWrapper):
    """AllenNLP compatible seq to box module"""

    def __init__(self,
                 module: torch.nn.modules.RNNBase,
                 box_type='SigmoidBoxes') -> None:

        if module.hidden_size % 2 != 0:
            raise ValueError(
                "module.hidden_size  has to be even but is {}".format(
                    module.hidden_size))

        if not module.batch_first:
            raise ValueError("module.batch_first should be True")
        super().__init__(module)

        if isinstance(module, torch.nn.LSTM):
            if box_type not in  ['TanhActivatedBoxes','TanhActivatedCenterSideBoxes','TanhActivatedMinMaxBoxTensor']:
                raise ValueError("Can only use TanhActivated* boxes with torch.nn.LSTM as encoder. But found {}".format(box_type))

        self.box_type = box_type
        self.boxes = BoxView(box_type, split_dim=-1)

    def get_output_dim(self, after_box: bool = False) -> int:
        """

        .. todo:: Logically correct output for get_output_dim when output is boxes
        """

        if after_box:
            return int(super().get_output_dim() /
                       2)  # this is still not right becuase
        # for boxes, last two dims are their representation
        else:
            return super().get_output_dim()

    def forward(self,
                inp: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                hidden_state: Optional[torch.Tensor] = None) -> str:
        output = super().forward(
            inp, mask, hidden_state)  # shape = (batch, hidden_size*num_dir)

        # when LSTM is bidirectional, the output of both directions is used in
        # z as well as Z. Hence, output of both directions is split

        if self._module.bidirectional:

            if self._module.batch_first:
                batch = output.size(0)
                split_on_dir = output.view(batch, 2, self._module.hidden_size)
                output_dir1 = split_on_dir[..., 0, :]
                output_dir2 = split_on_dir[..., 1, :]

            boxes_dir1 = self.boxes(output_dir1)
            boxes_dir2 = self.boxes(output_dir2)
            box_output = (self.boxes.box_types[self.box_type]).cat(
                (boxes_dir1, boxes_dir2))
        else:
            box_output: TBoxTensor = self.boxes(output)

        return box_output


class BoxEmbedding(Embedding):
    box_types = {'SigmoidBoxTensor': SigmoidBoxTensor, 'DeltaBoxTensor': DeltaBoxTensor, 'BoxTensor': BoxTensor}

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def __init__(
            self,
            num_embeddings: int,
            box_embedding_dim: int,
            box_type='SigmoidBoxTensor',
            weight: torch.FloatTensor = None,
            padding_index: int = None,
            trainable: bool = True,
            max_norm: float = None,
            norm_type: float = 2.0,
            scale_grad_by_freq: bool = False,
            sparse: bool = False,
            vocab_namespace: str = None,
            pretrained_file: str = None,
    ) -> None:
        """Similar to allennlp embeddings but returns box
        tensor by splitting the output of usual embeddings
        into z and Z

        Arguments:
            box_embedding_dim: Embedding weight would be box_embedding_dim*2
        """
        super().__init__(
            num_embeddings,
            box_embedding_dim * 2,
            weight=weight,
            padding_index=padding_index,
            trainable=trainable,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            vocab_namespace=vocab_namespace,
            pretrained_file=pretrained_file)
        try:
            self.box = self.box_types[box_type]
        except KeyError as ke:
            raise ValueError("Invalid box type {}".format(box_type)) from ke
        self.box_embedding_dim = box_embedding_dim
        self.init_weights()

    def forward(self, inputs: torch.LongTensor):
        emb = super().forward(inputs)  # shape (**, self.box_embedding_dim*2)
        box_emb = self.box.from_split(emb)

        return box_emb
