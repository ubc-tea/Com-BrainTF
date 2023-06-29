import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel
import pickle


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, nHead=4, local_transformer=False):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=nHead,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.local_transformer = local_transformer
        if local_transformer:
            self.pooling = False
        else:
            self.pooling = pooling
        if self.pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

        if local_transformer:
            self.class_token = nn.ParameterList()
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda())
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda())
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda())
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda())
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda())
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda())
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda())
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda())
        self.reset_parameters(local_transformer)

        self.mlp = nn.Sequential(
            nn.Linear(8*input_feature_size, 1024),
            nn.Linear(1024, input_feature_size),
            nn.ReLU()
        )       

    def reset_parameters(self, local_transformer=False):
        if local_transformer:
            for i in range(len(self.class_token)):
                self.class_token[i] = nn.init.xavier_normal_(self.class_token[i])
        

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, 
            x: torch.tensor, cluster_num=-1):
        bz, node_num, dim = x.shape
        if self.local_transformer:
            class_token = self.class_token[cluster_num]
            class_token = class_token.repeat(bz,1,1)
            x = torch.cat((class_token, x), dim=1)
        x = self.transformer(x)
        if self.local_transformer:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            return x, None, cls_token.reshape(x.shape[0], 1, -1)
        else:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            if self.pooling:
                x, assignment = self.dec(x)
                return x, assignment, cls_token.reshape(x.shape[0], 1, -1)
            else:
                return x, None, cls_token.reshape(x.shape[0], 1, -1)

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class ComBrainTF(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        self.num_MHSA = config.model.num_MHSA
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling 

        self.local_transformer = TransPoolingEncoder(input_feature_size=forward_dim,
                                                     input_node_num=in_sizes[1],
                                                     hidden_size=1024,
                                                     output_node_num=sizes[1],
                                                     pooling=False,
                                                     orthogonal=config.model.orthogonal,
                                                     freeze_center=config.model.freeze_center,
                                                     project_assignment=config.model.project_assignment,
                                                     nHead=config.model.nhead,
                                                     local_transformer=True)

        if config.model.num_MHSA == 1:
                self.attention_list.append(
                    TransPoolingEncoder(input_feature_size=forward_dim,
                                        input_node_num=in_sizes[1],
                                        hidden_size=1024,
                                        output_node_num=sizes[1],
                                        pooling=do_pooling[1],
                                        orthogonal=config.model.orthogonal,
                                        freeze_center=config.model.freeze_center,
                                        project_assignment=config.model.project_assignment,
                                        nHead=config.model.nhead,
                                        local_transformer=False))
        else:
            for index, size in enumerate(sizes):
                self.attention_list.append(
                    TransPoolingEncoder(input_feature_size=forward_dim,
                                        input_node_num=in_sizes[index],
                                        hidden_size=1024,
                                        output_node_num=size,
                                        pooling=do_pooling[index],
                                        orthogonal=config.model.orthogonal,
                                        freeze_center=config.model.freeze_center,
                                        project_assignment=config.model.project_assignment,
                                        nHead=config.model.nhead,
                                        local_transformer=False))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        self.assignMat = None
        self.mlp = nn.Sequential(
            nn.Linear(8 * forward_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, forward_dim),
            nn.LeakyReLU()
        )

        with open('node_clus_map.pickle', 'rb') as handle:
            self.node_clus_map = pickle.load(handle)

        self.node_rearranged_len = [41, 70, 91, 110, 130, 137, 158, 200]

    def rearrange_node_feature(self, node_feature_rearranged, node_feature, rearranged_indices):
        # Rearrange according to node_clus_map which is a dictionary {0:1, 1:3, .... 199:7}
        node_feature_rearranged = node_feature[:, rearranged_indices, :]
        node_feature_rearranged = node_feature_rearranged[:, :, rearranged_indices]
        return node_feature_rearranged

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):

        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []
        attn_weights = []

        node_feature_rearranged = torch.tensor(node_feature.shape).cuda()
        node_feature_rearranged = self.rearrange_node_feature(node_feature_rearranged, node_feature, list(self.node_clus_map.keys()))

        node_feature_rearranged[:,:self.node_rearranged_len[0], :], _, local_class_tokens0  = self.local_transformer(node_feature_rearranged[:, :self.node_rearranged_len[0], :], cluster_num = 0)
        node_feature_rearranged[:,self.node_rearranged_len[0]:self.node_rearranged_len[1], :], _, local_class_tokens1 = self.local_transformer(node_feature_rearranged[:, self.node_rearranged_len[0]:self.node_rearranged_len[1], :], cluster_num = 1)
        node_feature_rearranged[:,self.node_rearranged_len[1]:self.node_rearranged_len[2], :], _, local_class_tokens2 = self.local_transformer(node_feature_rearranged[:, self.node_rearranged_len[1]:self.node_rearranged_len[2], :], cluster_num = 2)
        node_feature_rearranged[:,self.node_rearranged_len[2]:self.node_rearranged_len[3], :], _, local_class_tokens3 = self.local_transformer(node_feature_rearranged[:, self.node_rearranged_len[2]:self.node_rearranged_len[3], :], cluster_num = 3)
        node_feature_rearranged[:,self.node_rearranged_len[3]:self.node_rearranged_len[4], :], _, local_class_tokens4 = self.local_transformer(node_feature_rearranged[:, self.node_rearranged_len[3]:self.node_rearranged_len[4], :], cluster_num = 4)
        node_feature_rearranged[:,self.node_rearranged_len[4]:self.node_rearranged_len[5], :], _, local_class_tokens5 = self.local_transformer(node_feature_rearranged[:, self.node_rearranged_len[4]:self.node_rearranged_len[5], :], cluster_num = 5)
        node_feature_rearranged[:,self.node_rearranged_len[5]:self.node_rearranged_len[6], :], _, local_class_tokens6 = self.local_transformer(node_feature_rearranged[:, self.node_rearranged_len[5]:self.node_rearranged_len[6], :], cluster_num = 6)
        node_feature_rearranged[:,self.node_rearranged_len[6]:self.node_rearranged_len[7], :], _, local_class_tokens7 = self.local_transformer(node_feature_rearranged[:, self.node_rearranged_len[6]:self.node_rearranged_len[7], :], cluster_num = 7)

        node_feature = node_feature_rearranged
        class_token = torch.cat((local_class_tokens0,local_class_tokens1,local_class_tokens2,local_class_tokens3,local_class_tokens4,
                                     local_class_tokens5,local_class_tokens6,local_class_tokens7), dim=1)
        class_token = class_token.reshape((bz, -1))
        class_token = self.mlp(class_token)
        class_token = class_token.reshape((bz, 1, -1))
        node_feature = torch.cat((class_token, node_feature), dim=1)   
        
        if self.num_MHSA == 1:
            node_feature, assign_mat, cls_token = self.attention_list[0](node_feature)
            assignments.append(assign_mat)
            attn_weights.append(self.attention_list[0].get_attention_weights())
        else:
            for atten in self.attention_list:
                node_feature, _, cls_token = atten(node_feature)
                attn_weights.append(atten.get_attention_weights())

        self.assignMat = assignments[0]

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature), None

    def get_assign_mat(self):
        return self.assignMat

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_local_attention_weights(self):
        return self.local_transformer.get_attention_weights()

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
