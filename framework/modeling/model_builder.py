import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class NaiveModelBuilder(nn.Module):
    def __init__(self, model_s, model_t):
        super(NaiveModelBuilder, self).__init__()
        self.model_s = model_s
        self.model_t = model_t

        # freeze the teacher model
        for p in self.model_t.parameters():
            p.requires_grad = False

    def forward(self, input):
        output_s, mat_s = self.model_s(input)
        if self.training == False:
            return output_s

        output_t, mat_t = self.model_t(input)
        return output_s, output_t


class ModelBuilder(nn.Module):
    def __init__(self, model_s, model_t, model_kd, normalized, preReLU):
        super(ModelBuilder, self).__init__()
        self.model_s = model_s
        self.model_t = model_t
        self.model_kd = model_kd
        self.normalized = normalized
        self.preReLU = preReLU

        # freeze the teacher model
        for p in self.model_t.parameters():
            p.requires_grad = False

    def forward(self, input):
        output_s, mat_s = self.model_s(input, preReLU=self.preReLU)
        if self.training == False:
            return output_s

        _, mat_t = self.model_t(input, preReLU=self.preReLU)
        ret_s, ret_t = self.model_kd(mat_s, mat_t)
        if self.normalized:
            normal_s = []
            normal_t = []
            num_kd = len(ret_s)
            for j in range(num_kd):
                B = ret_s[j].size(0)
                l2_norm = torch.norm(ret_s[j].view(B, -1), p=2, dim=1).view(B, 1, 1, 1)
                normal_s.append(ret_s[j] / 100)
                l2_norm = torch.norm(ret_t[j].view(B, -1), p=2, dim=1).view(B, 1, 1, 1)
                # normal_t.append(ret_t[j] / l2_norm)
            ret_s = normal_s
            ret_t = normal_t
        return output_s, ret_s, ret_t

