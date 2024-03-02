import os
from os.path import join as ospj
import json
import cv2 as cv


from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    # if isinstance(module, nn.Conv2d):
    #     nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    #     if module.bias is not None:
    #         nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)#返回一个内容全为1,size为1×C×H×W的张量
    x_src_with_wb = torch.cat([wb, x_src], dim=0)#拼接为一个N+1,C,H,W的张量
    print(y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    a_src = nets.anencoder(x_src)
    a_trg = nets.anencoder(x_ref)
    s_src = nets.style_encoder(x_src, None)
    s_ref = nets.style_encoder(x_ref, None)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]#Tensor变list
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(a_src, s_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]
    x_nosty = nets.generator(a_src, None, masks=masks)
    x_nosty_with_web = torch.cat([wb, x_nosty], dim=0)
    x_concat += [x_nosty_with_web]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)

    del x_concat

def translate_using_onereference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)#返回一个内容全为1,size为1×C×H×W的张量
    x_src_with_wb = torch.cat([wb, x_src], dim=0)#拼接为一个N+1,C,H,W的张量

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    a_src = nets.anencoder(x_src)
    a_trg = nets.anencoder(x_ref)
    s_src = nets.style_encoder(x_src, None)
    s_ref = nets.style_encoder(x_ref, None)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]#Tensor变list
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(a_src, s_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]
    x_nosty = nets.generator(a_src, None, masks=masks)
    x_nosty_with_web = torch.cat([wb, x_nosty], dim=0)
    x_concat += [x_nosty_with_web]
    x_concat = torch.cat(x_concat, dim=0)
    # save_image(x_concat, N+1, filename)
    i = 1
    for tensor in x_concat:
        filename = ospj(args.result_dir, '%02d_reference.png'% (i))
        save_image(tensor,1,filename)
        i=i+1
    del x_concat

@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    device = inputs.x_src.device
    N = inputs.x_src.size(0)

    # translate and reconstruct (reference-guided)
    # filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    # translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided image synthesis
    # y_trg_list = [torch.tensor(y).repeat(N).to(device)
    #               for y in range(min(args.num_domains, 5))]
    # z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    # for psi in [0.5, 0.7, 1.0]:
    #     filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
    #     translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.png' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@torch.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, args, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo', 
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255

def unnormalize(tensor):
    # 假设已经定义了mean和std
    min_val = tensor.min()
    max_val = tensor.max()
    # 定义transforms.Normalize()的逆操作
    output_tensor = (tensor-min_val)/(max_val-min_val)

    return output_tensor


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def sce_criterion(logits, labels):
    logits = logits.view(logits.size(0), logits.size(1), -1).mean(dim=2)

        # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)

    return loss

import torch
import cv2
import numpy as np
from PIL import Image

def segment_objects(image_tensor, threshold=30):
    # 将三通道图像张量转为单通道灰度图像张量
    gray_tensor = torch.mean(image_tensor, dim=1, keepdim=True)

    # 将PyTorch张量转换为NumPy数组
    gray_np = gray_tensor.permute(0, 2, 3, 1).cpu().numpy()

    segmented_images = []

    for img in gray_np:
        # 确保灰度图的大小与原始图像相同
        mask = cv2.resize((img[:, :, 0] > threshold).astype(np.uint8), (img.shape[1], img.shape[0]))

        # 位与运算，保留原始颜色在mask非零的位置
        segmented_image = cv2.bitwise_and(img, img, mask=mask)

        segmented_images.append(segmented_image)

    # 将分割后的图像列表转换为PyTorch张量
    segmented_tensor = torch.from_numpy(np.stack(segmented_images)).permute(0, 3, 1, 2).to(image_tensor.device)

    return segmented_tensor


def segmentation(images, threshold=0):

    batch_size = images.size(0)
    results = torch.zeros((batch_size, 1, 256, 256), dtype=torch.float32)

    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)  # 将 tensor 转为 numpy 数组，并调整维度顺序
        # print('image:',image.shape)
        # 进行第一次阈值分割，得到脑部区域掩膜
        _, brain_mask = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)


        # 对脑部区域掩膜进行按位与操作，得到脑部图像
        brain_image = cv.bitwise_and(brain_mask, image)

        brain_image = brain_image[:, :, 0]
        results[i, 0] = torch.from_numpy(brain_image).float() / 255.0  # 将 numpy 数组转为 tensor，并将像素值归一化到 [0, 1]
 
    return images

def save_segmented_images(image_tensor, output_path, threshold=30):
    # 分割图像
    segmented_result = segment_objects(image_tensor, threshold)

    # 转换为可保存的图像格式（HWC）
    segmented_images = [segmented_result[i].permute(1, 2, 0).cpu().numpy() for i in range(segmented_result.shape[0])]

    # 保存分割后的图像
    for i, segmented_image in enumerate(segmented_images):
        image_save_path = f"{output_path}/segmented_image_{i+1}.png"
        cv2.imwrite(image_save_path, cv2.cvtColor((segmented_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"Segmented image saved at: {image_save_path}")
