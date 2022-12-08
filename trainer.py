import torch
from networks.discriminator import Discriminator
from networks.generator import ExpGenerator
import torch.nn.functional as F
from torch import nn, optim
import os
from vgg19 import VGGLoss
from torch.nn.parallel import DistributedDataParallel as DDP


def requires_grad(net, flag=True, query=[]):
    for name, p in net.named_parameters():
        if len(query) > 0:
            for q in query:
                if q in name:
                    p.requires_grad = flag
        else:
            p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device, rank):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size

        self.gen = ExpGenerator(args.size, args.latent_dim_style, args.latent_dim_motion, args.exp_dim, args.channel_multiplier).to(
            device)
        self.dis = Discriminator(args.size, args.channel_multiplier).to(device)

        # distributed computing
        self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.gen.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )

        self.criterion_vgg = VGGLoss().to(rank)

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def uniform_loss(self, gen):
        loss_weight = 50
        unif_loss = lambda x: -torch.log((1 + x).clamp(min=1e-6)).mean()
        src_exp_code = gen['h_exp_src'] # B x motion_dim
        drv_exp_code = gen['h_exp_drv'] # B x motion_dim
        greater_mask = (src_exp_code >= drv_exp_code).detach() 
        less_mask = ~greater_mask
        greater_labels = torch.cat([src_exp_code[greater_mask], drv_exp_code[less_mask]], dim=0)
        less_labels = torch.cat([src_exp_code[less_mask], drv_exp_code[greater_mask]], dim=0)
        loss =  loss_weight * (unif_loss(greater_labels) + unif_loss(-less_labels))
        return loss


    def kd_motion_loss(self, gen):
        loss_weight = 1
        h_motion = gen['h_motion']
        h_motion_tf  = gen['h_motion_tf']
        return F.l1_loss(h_motion, h_motion_tf)

    def gen_update(self, img_source, img_target):
        self.gen.train()
        self.gen.zero_grad()

        requires_grad(self.gen, True, query=['exp'])
        requires_grad(self.dis, False)

        # img_target_recon = self.gen(img_source, img_target)
        # img_recon_pred = self.dis(img_target_recon)

        gen_res = self.gen(img_source, img_target)
        img_target_recon = gen_res['img_recon']
        img_recon_pred = self.dis(img_target_recon)

        vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
        l1_loss = F.l1_loss(img_target_recon, img_target)
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)
        unif_loss = self.uniform_loss(gen_res)
        kd_loss = self.kd_motion_loss(gen_res)

        g_loss = vgg_loss + l1_loss + gan_g_loss + unif_loss + kd_loss

        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, gan_g_loss, img_target_recon, unif_loss, kd_loss

    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        real_img_pred = self.dis(img_real)
        recon_img_pred = self.dis(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, img_source, img_target):
        with torch.no_grad():
            self.gen.eval()

            img_recon = self.gen(img_source, img_target)['img_recon']
            img_source_ref = self.gen(img_source, None)['img_recon']

        return img_recon, img_source_ref

    def load_ckpt(self, net, ckpt, name):
        if name in ckpt:
            ckpt = ckpt[name]
            state_dict = net.state_dict()
            state_dict.update(ckpt)
            net.load_state_dict(state_dict)
        else:
            print(f'name ({name}) not exists in checkpoint')

    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt)
        ckpt_name = os.path.basename(resume_ckpt)
        start_iter = int(os.path.splitext(ckpt_name)[0])


        # self.gen.module.load_state_dict(ckpt["gen"])
        # self.dis.module.load_state_dict(ckpt["dis"])
        # self.g_optim.load_state_dict(ckpt["g_optim"])
        # self.d_optim.load_state_dict(ckpt["d_optim"])

        self.load_ckpt(self.gen.module, ckpt, "gen")
        self.load_ckpt(self.dis.module, ckpt, "dis")
        # self.load_ckpt(self.g_optim, ckpt, "g_optim")
        # self.load_ckpt(self.d_optim., ckpt, "d_optim")

        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "dis": self.dis.module.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
