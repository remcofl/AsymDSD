from torch import nn


class IdentityMultiArg(nn.Identity):
    def forward(self, arg, **kwargs):
        return arg


class IdentityPassThrough(nn.Identity):
    def forward(self, *args):
        return args
