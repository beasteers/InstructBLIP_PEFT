"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast



from torch.autograd.function import Function
class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class TFQFormerClassHead(nn.Module):
    def __init__(self, input_size, n_classes, num_heads=32, num_encoder_layers=1, seq_len=32, hidden_act='gelu'):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_size) * 0.02)
        # # self.cls_token = torch.randn(1, 1, input_size).cuda()
        # # self.cls_token = torch.ones(1, 1, input_size).cuda()
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len+1, input_size))  # Assuming max seq length + cls token
        # self.pos_encoder = PositionalEncoding(input_size, seq_len+1)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, activation=hidden_act)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)
        # self.attn = nn.MultiheadAttention(input_size, num_heads)
        # self.fn1 = nn.Linear(input_size, input_size)
        self.fn2 = nn.Linear(input_size, n_classes)
        # self.act = nn.GELU()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, tokens, targets=None):
        # tokens = _ScaleGradient.apply(tokens, 1/5)
        x = tokens
        # tokens = self.fn1(tokens)
        # tokens = self.act(tokens)
        # logits = tokens


        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # cls_tokens = x.mean(1, keepdims=1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encodings
        x += self.pos_encoder[:, :x.size(1), :]
        # x = x[:, ::-1]
        x = self.transformer_encoder(x)
        # x, attn_weights = self.attn(x, x, x)
        # x = self.act(x)
        # print(x.shape)
        x = x[:, 0]
        # x = x.mean(1)
        # print(x.shape)
        logits = self.fn2(x)
        # print(logits.shape)

        loss = 0
        if targets is not None:
            loss = self.loss(logits, targets.float())
            loss = (loss[targets >= 0]).sum() / max(1, (targets >= 0).sum())
        # print(loss)
        y = torch.sigmoid(logits)
        # print(y)
        return {
            'loss': loss,
            'prediction': y,
        }


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe[None])

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)


# class QFormerClassHead(nn.Module):
#     def __init__(self, input_size, n_classes, seq_len=32):
#         super().__init__()
#         # self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, input_size))
#         self.fn1 = nn.Linear(input_size, n_classes)
#         # self.act = nn.GELU()
#         self.loss = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, tokens, targets=None):
#         x = tokens
#         # x = x[:, 0]
#         x = x.mean(1)
#         logits = self.fn1(x)

#         loss = 0
#         if targets is not None:
#             loss = self.loss(logits, targets.float())
#             loss = (loss[targets >= 0]).sum() / max(1, (targets >= 0).sum())
#         y = torch.sigmoid(logits)
#         return {
#             'loss': loss,
#             'prediction': y,
#         }


class MHAQFormerClassHead(nn.Module):
    def __init__(self, input_size, n_classes, num_heads=32, num_encoder_layers=1, seq_len=32, hidden_act='gelu'):
        super().__init__()
        # self.cls_token = nn.Parameter(torch.randn(1, 1, input_size) * 0.02)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len+1, input_size))  # Assuming max seq length + cls token
        self.attn = nn.MultiheadAttention(input_size, num_heads)
        self.fn2 = nn.Linear(input_size, n_classes)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, tokens, targets=None):
        x = tokens
        # cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # cls_tokens = x.mean(1, keepdims=1)
        # x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encodings
        x += self.pos_encoder[:, :x.size(1), :]
        x, attn_weights = self.attn(x, x, x)
        x = x[:, 0]
        # x = x.mean(1)
        logits = self.fn2(x)

        loss = 0
        if targets is not None:
            loss = self.loss(logits, targets.float())
            loss = (loss[targets >= 0]).sum() / max(1, (targets >= 0).sum())
        y = torch.sigmoid(logits)
        return {
            'loss': loss,
            'prediction': y,
        }
        


class DenseQFormerClassHead(nn.Module):
    def __init__(self, input_size, n_classes, seq_len=32):
        super().__init__()
        # self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, input_size))
        # self.fn1 = nn.Linear(input_size, input_size)
        self.fn2 = nn.Linear(input_size, n_classes)
        # self.act = nn.ReLU()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, tokens, targets=None):
        x = tokens
        # x = x[:, 0]
        # x = self.fn1(x)
        # x = self.act(x)
        x = x.mean(1)
        logits = self.fn2(x)

        loss = 0
        if targets is not None:
            loss = self.loss(logits, targets.float())
            loss = (loss[targets != -1]).sum() / max(1, (targets != -1).sum())
            # print(loss)
        y = torch.sigmoid(logits)
        # if targets is not None:
        #     print(torch.cat([torch.round(logits, decimals=3), targets, torch.round(y, decimals=3)], dim=1))
        return {
            'loss': loss,
            'prediction': y,
        }


CLS_HEAD = {
    'dense': DenseQFormerClassHead,
    'mha': MHAQFormerClassHead,
    'transformer': TFQFormerClassHead,
    None: DenseQFormerClassHead
}