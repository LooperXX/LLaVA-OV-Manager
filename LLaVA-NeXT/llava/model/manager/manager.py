import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers.models.clip.modeling_clip import CLIPVisionModel

# Modified from https://github.com/LooperXX/ManagerTower/blob/main/src/modules/heads.py

class StaticManagerZeroInit(nn.Module):
    def __init__(self, config, num_manage_select_layer, layer_index, hidden_size):
        '''
        :param config:
        :param num_manage_select_layer: the number of layers to be managed
        :param layer_index: the index of the current layer injected by the manager
        :param hidden_size: the hidden size of the language model
        :param vision_hidden_size: the hidden size of the vision model
        '''
        super().__init__()
        self.config = config
        self.num_manage_select_layer = num_manage_select_layer
        self.layer_index = layer_index
        self.hidden_size = hidden_size
        if layer_index == 0:
            self.num_manage_select_layer -= 1

        # zero init self.layer_scores
        self.layer_scores = nn.Parameter(torch.zeros(1, self.num_manage_select_layer, 1, hidden_size), requires_grad=True)

    def aggregate_reps(self, layer_scores, layer_reps, is_training=False):
        # layer_scores: 1 x N x 1 x D
        # layer_reps: B x N x L x D

        if is_training:
            layer_scores = layer_scores * noise_jitter(layer_scores)

        return torch.sum(layer_scores * layer_reps, dim=1)

    def forward(self, hidden_states, cross_modal_hidden_states, extra_query=None, is_training=False):
        # hidden_states: B x N x L x D
        # cross_modal_hidden_states: B x L x D
        layer_scores_ = self.layer_scores
        if self.layer_index == 0:
            hidden_states = hidden_states[:, :-1]
        hidden_states = self.aggregate_reps(layer_scores_, hidden_states, is_training=is_training)

        return hidden_states


class AdaptiveManagerZeroInit(nn.Module):
    def __init__(self, config, num_manage_select_layer, layer_index, hidden_size):
        '''
        :param config:
        :param num_manage_select_layer: the number of layers to be managed
        :param layer_index: the index of the current layer injected by the manager
        :param hidden_size: the hidden size of the language model
        :param vision_hidden_size: the hidden size of the vision model
        '''
        super().__init__()
        self.config = config
        self.num_manage_select_layer = num_manage_select_layer
        self.layer_index = layer_index
        if layer_index == 0:
            raise NotImplementedError('Not implemented yet for the first layer')
        self.hidden_size = hidden_size

        # for extra query
        self.query_type = self.config.mm_manager_type.replace("_zerouni", "").split('adaptive_')[-1] if 'adaptive_' in self.config.mm_manager_type.replace("_zerouni", "") else 'none'
        if self.query_type in ['cross', 'last', 'average']:
            if self.query_type == 'cross':
                self.fusion_attention = LayerAttention(self.hidden_size, self.hidden_size)
            elif self.query_type in ['last', 'average']:
                pass
            else:
                raise NotImplementedError(f'Unknown query type {self.query_type}')
            self.linear_controller = nn.Linear(hidden_size * 2, num_manage_select_layer, bias=False) # B x 2D -> B x N
        else:
            self.linear_controller = nn.Linear(hidden_size, num_manage_select_layer, bias=False) # B x D -> B x N

        self.zero_gate = nn.Parameter(torch.zeros(1, self.num_manage_select_layer, 1, hidden_size), requires_grad=True)

    def aggregate_reps(self, layer_scores, layer_reps, is_training=False):
        # layer_scores: 1 x N x 1 x D
        # layer_reps: B x N x L x D

        if is_training:
            layer_scores = layer_scores * noise_jitter(layer_scores)

        return torch.sum(layer_scores * self.zero_gate * layer_reps, dim=1)

    def forward(self, hidden_states, cross_modal_hidden_states, extra_query=None, is_training=False):
        # hidden_states: B x N x L x D
        # cross_modal_hidden_states: B x L x D
        # extra_query: B x L x D
        if self.layer_index == 0:
            raise NotImplementedError('Not implemented yet for the first layer')

        if self.query_type != 'none':
            # B x L x 2D -> B x L x N => B x N x L x 1 => B x N x L x D
            if extra_query is None:
                print('extra_query is None !! use self as extra_query !!')
                extra_query = cross_modal_hidden_states

            if self.query_type == 'last':
                extra_query = extra_query[:, -1] # B x D, only use the last token representation
            elif self.query_type in ['average']:
                extra_query = torch.mean(extra_query, dim=1) # B x D, average all token representations

            if self.query_type == 'cross':
                extra_query = extra_query.expand(cross_modal_hidden_states.shape[0], -1, -1) # B x L x D
                fused_query = torch.cat((torch.softmax(self.fusion_attention(cross_modal_hidden_states, extra_query), dim=-1) @ extra_query , cross_modal_hidden_states), dim=-1) # B x L x 2D
                layer_scores_generate = self.linear_controller(fused_query).transpose(1, 2).unsqueeze(-1)
            elif self.query_type in ['last', 'average']:
                extra_query = extra_query.unsqueeze(1).expand(cross_modal_hidden_states.shape[0], hidden_states.shape[2], -1) # B x L x D
                fused_query = torch.cat((extra_query, cross_modal_hidden_states), dim=-1)
                layer_scores_generate = self.linear_controller(fused_query).transpose(1, 2).unsqueeze(-1)
        else:
            # B x L x D => B x L x N => B x N x L x 1 => B x N x L x D
            layer_scores_generate = self.linear_controller(cross_modal_hidden_states).transpose(1, 2).unsqueeze(-1)


        layer_scores_generate = layer_scores_generate.expand(-1, -1, -1, self.hidden_size)

        hidden_states = self.aggregate_reps(layer_scores_generate, hidden_states, is_training=is_training)

        return hidden_states

def noise_jitter(x, epsilon=1e-2):
    low: float = 1.0 - epsilon
    high: float = 1.0 + epsilon
    noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
    return low + noise * (high - low)

class LayerAttention(nn.Module):
    def __init__(self, query_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(query_dim, self.hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.hidden_size)

    def forward(self, query, key):
        q = self.q_proj(query)
        k = self.k_proj(key)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        return attention_scores
