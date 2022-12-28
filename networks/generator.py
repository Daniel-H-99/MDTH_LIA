from torch import nn
from .encoder import ExpSeqEncoder
from .styledecoder import Synthesis

class ExpGenerator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, exp_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(ExpGenerator, self).__init__()

        # encoder
        self.enc = ExpSeqEncoder(size, style_dim, motion_dim, exp_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def encode(self, img_source, img_drive, img_prev, img_next):
        res = {}
        enc = self.enc(img_source, img_drive, img_prev, img_next)
        res.update(enc)
        return res

    def forward(self, img_source, img_drive, img_prev, img_next, h_exp=None, noise=None, dropout=0):
        res = {}
        # img_drive = img_source
        # img_prev = img_source
        # img_next = img_source
        enc = self.enc(img_source, img_drive, img_prev, img_next, h_exp=h_exp, noise=noise, dropout=dropout)
        res.update(enc)
        wa = enc['h_source']
        alpha = enc['h_motion']
        if alpha is not None:
            alpha = [alpha]
        feats = enc['feats']
        img_recon = self.dec(wa, alpha, feats)
        res['img_recon'] = img_recon
        return res

class Generator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def forward(self, img_source, img_drive, h_start=None):
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        img_recon = self.dec(wa, alpha, feats)

        return img_recon

