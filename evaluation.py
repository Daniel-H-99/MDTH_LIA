import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from dataset import Vox256_eval, Taichi_eval, TED_eval
from torch.utils import data
from PIL import Image
# import lpips
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32
import imageio
import cv2
from scipy.spatial import ConvexHull


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def save_video(save_path, name, vid_target_recon, fps=10.0):
    vid = (vid_target_recon.permute(0, 2, 3, 4, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    torchvision.io.write_video(save_path + '%s.mp4' % name, vid[0].cpu(), fps=fps)


def data_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm

def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device= 'cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except:
            pass
    return frame_num

class EvaPipeline(nn.Module):
    def __init__(self, args):
        super(EvaPipeline, self).__init__()

        self.args = args

        transform = torchvision.transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )

        # if args.dataset == 'vox':
        #     path = 'checkpoints/vox.pt'
        #     dataset = Vox256_eval(transform)
        # elif args.dataset == 'taichi':
        #     path = 'checkpoints/taichi.pt'
        #     dataset = Taichi_eval(transform)
        # elif args.dataset == 'ted':
        #     path = 'checkpoints/ted.pt'
        #     dataset = TED_eval(transform)
        # else:
        #     raise NotImplementedError

        # os.makedirs(os.path.join(self.save_path, args.dataset), exist_ok=True)

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        # print('==> loading data')
        # self.loader = data.DataLoader(
        #     dataset,
        #     num_workers=1,
        #     batch_size=1,
        #     drop_last=False,
        # )

        # self.loss_fn = lpips.LPIPS(net='alex').cuda()

    def inference(self, opt, save_frames, driving_exps=None, driving_exps_mask=None):
        device = 'cuda:0'
        bs = 1
        transform = torchvision.transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
        
        def get_frame(path):
            x = Image.open(path).convert('RGB')
            x = transform(x)
            x = x[None].to(device)
            return x

        
        driving_frames_path = os.listdir(os.path.join(opt.driving_dir, 'frames'))
        driving_video = []
        fids = []
        for frame_path in driving_frames_path:
            driving_frame = get_frame(os.path.join(opt.driving_dir, 'frames', frame_path))
            driving_video.append(driving_frame)
            fid = int(frame_path.split('.png')[0])
            fids.append(fid)
            
        order = torch.tensor(fids).argsort()
        driving_video = torch.cat(driving_video, dim=0)[order]
        
        if opt.source_dir.endswith('.mp4'):
            source_frames = sorted(os.listdir(os.path.join(opt.source_dir, 'frames')))
            random_idx = len(source_frames) // 2
            source_image = get_frame(os.path.join(opt.source_dir, 'frames', source_frames[random_idx]))
            # source_image = get_frame(os.path.join(opt.source_dir, 'frames', '0000000.png'))
        else:
            source_image = get_frame(os.path.join(opt.source_dir, 'image.png'))

        # raw_scale_path = os.path.join(opt.source_dir, 'scale.txt')
        # raw_scale = np.loadtxt(raw_scale_path, dtype=np.float32, delimiter=',', comments=None)
        # frame_shape = source_image.shape[2:]
        # if raw_scale[0] > raw_scale[1]:
        #     final_shape = (frame_shape[0] * raw_scale[1] / raw_scale[0], frame_shape[0])
        # else:
        #     final_shape = (frame_shape[0], frame_shape[0] * raw_scale[0] / raw_scale[1])
        # final_shape = (int(final_shape[0]), int(final_shape[1]))

        # if len(source_image.shape) == 2:
        #     source_image = cv2.cvtColor(source_image, cv2.COLOR_GRAY2RGB)

        frame_shape = (256, 256, 3)
        
        # source_image = frame_transform(source_image)
        
        # source_image = resize(img_as_float32(source_image), frame_shape[:2])[..., :3]
        # source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)
        # source = source.to(device)
        source = source_image
        
        predictions = []

        relative_movement = False

        if relative_movement:
            source_cpu = source.detach().cpu()[0].permute(1, 2, 0).clamp(0, 1).numpy()
            driving_cpu = driving_video.detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
            # i = find_best_frame(source_cpu, driving_cpu, False)
            i = 0
            print ("Best frame: " + str(i))
            h_start = self.gen.enc.enc_motion(driving_video[[i], :, :, :])

            for frame_idx in tqdm(range(0, len(driving_video), bs)):
                driving_frame = driving_video[frame_idx:frame_idx+bs].to(device)
                if len(driving_frame) < bs:
                    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).repeat(len(kp_driving['value']), 1, 1, 1)
                    source = source.to(device)
                
                prediction = self.gen(source, driving_frame, h_start)
                predictions.append(np.transpose(prediction.data.cpu().numpy(), [0, 2, 3, 1]))
        else:
            for frame_idx in tqdm(range(0, len(driving_video), bs)):
                driving_frame = driving_video[frame_idx:frame_idx+bs].to(device)
                if len(driving_frame) < bs:
                    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).repeat(len(kp_driving['value']), 1, 1, 1)
                    source = source.to(device)
                
                prediction = self.gen(source, driving_frame)
                predictions.append(np.transpose(prediction.data.cpu().numpy(), [0, 2, 3, 1]))

        vid = np.concatenate(predictions, axis=0).clip(-1, 1)
        vid = (vid - vid.min()) / (vid.max() - vid.min())

        # print(f'1: vid shape: {vid.shape}')
        meshed_frames = []
        # scaled_frames = []
        for i, frame in enumerate(vid):
            frame = np.ascontiguousarray(img_as_ubyte(frame))
            # if i >= len(target_meshes):
            #     continue
            # mesh = target_meshes[i]
            # frame = draw_section(mesh[:, :2].numpy().astype(np.int32), frame_shape, section_config=[OPENFACE_LEFT_EYEBROW_IDX, OPENFACE_RIGHT_EYEBROW_IDX, OPENFACE_NOSE_IDX, OPENFACE_LEFT_EYE_IDX, OPENFACE_RIGHT_EYE_IDX, OPENFACE_OUT_LIP_IDX, OPENFACE_IN_LIP_IDX] , mask=frame)
            meshed_frames.append(frame)
            # scaled_frames.append(cv2.resize(frame, final_shape))

        vid = meshed_frames
        # print(f'2: vid shape: {vid.shape}')

        imageio.mimsave(os.path.join(opt.result_dir, opt.result_video), vid, fps=25)
        # imageio.mimsave(os.path.join(opt.result_dir, 'scaled.mp4'), scaled_frames, fps=25)
        
        if save_frames:
            for i, frame in enumerate(vid):
                imageio.imwrite(os.path.join(opt.result_dir, 'frames', '{:05d}.png'.format(i)), frame)
                
        return predictions
                
    def run(self):

        loss_list = []
        loss_lpips = []
        for idx, (vid_name, vid) in tqdm(enumerate(self.loader)):

            with torch.no_grad():

                vid_real = []
                vid_recon = []
                img_source = vid[0].cuda()
                for img_target in vid:
                    img_target = img_target.cuda()
                    img_recon = self.gen(img_source, img_target)
                    vid_recon.append(img_recon.unsqueeze(2))
                    vid_real.append(img_target.unsqueeze(2))

                vid_recon = torch.cat(vid_recon, dim=2)
                vid_real = torch.cat(vid_real, dim=2)

                loss_list.append(torch.abs(0.5 * (vid_recon.clamp(-1, 1) - vid_real)).mean().cpu().numpy())
                vid_real = vid_real.permute(0, 2, 1, 3, 4).squeeze(0)
                vid_recon = vid_recon.permute(0, 2, 1, 3, 4).squeeze(0)
                loss_lpips.append(self.loss_fn.forward(vid_real, vid_recon.clamp(-1, 1)).mean().cpu().detach().numpy())

        print("reconstruction loss: %s" % np.mean(loss_list))
        print("lpips loss: %s" % np.mean(loss_lpips))


class Eva(nn.Module):
    def __init__(self, args):
        super(Eva, self).__init__()

        self.args = args

        transform = torchvision.transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )

        if args.dataset == 'vox':
            path = 'checkpoints/vox.pt'
            dataset = Vox256_eval(transform)
        elif args.dataset == 'taichi':
            path = 'checkpoints/taichi.pt'
            dataset = Taichi_eval(transform)
        elif args.dataset == 'ted':
            path = 'checkpoints/ted.pt'
            dataset = TED_eval(transform)
        else:
            raise NotImplementedError

        os.makedirs(os.path.join(self.save_path, args.dataset), exist_ok=True)

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')
        self.loader = data.DataLoader(
            dataset,
            num_workers=1,
            batch_size=1,
            drop_last=False,
        )

        self.loss_fn = lpips.LPIPS(net='alex').cuda()

    def run(self):

        loss_list = []
        loss_lpips = []
        for idx, (vid_name, vid) in tqdm(enumerate(self.loader)):

            with torch.no_grad():

                vid_real = []
                vid_recon = []
                img_source = vid[0].cuda()
                for img_target in vid:
                    img_target = img_target.cuda()
                    img_recon = self.gen(img_source, img_target)
                    vid_recon.append(img_recon.unsqueeze(2))
                    vid_real.append(img_target.unsqueeze(2))

                vid_recon = torch.cat(vid_recon, dim=2)
                vid_real = torch.cat(vid_real, dim=2)

                loss_list.append(torch.abs(0.5 * (vid_recon.clamp(-1, 1) - vid_real)).mean().cpu().numpy())
                vid_real = vid_real.permute(0, 2, 1, 3, 4).squeeze(0)
                vid_recon = vid_recon.permute(0, 2, 1, 3, 4).squeeze(0)
                loss_lpips.append(self.loss_fn.forward(vid_real, vid_recon.clamp(-1, 1)).mean().cpu().detach().numpy())

        print("reconstruction loss: %s" % np.mean(loss_list))
        print("lpips loss: %s" % np.mean(loss_lpips))


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, choices=['vox', 'taichi', 'ted'], default='')
    parser.add_argument("--save_path", type=str, default='./evaluation_res')
    args = parser.parse_args()

    demo = Eva(args)
    demo.run()
