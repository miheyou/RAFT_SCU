import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    # 检查输入张量是否在GPU上，如果在则先移到CPU处理插值
    if image1.is_cuda:
        image1_cpu = image1.cpu()
        image2_cpu = image2.cpu()
        
        # 在CPU上进行插值
        img1 = F.interpolate(image1_cpu, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2_cpu, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
        
        # 移回GPU
        img1 = img1.to(image1.device)
        img2 = img2.to(image2.device)
    else:
        img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    
    # 同样处理下采样
    if flow.is_cuda:
        flow_cpu = flow.cpu()
        info_cpu = info.cpu()
        
        flow_down = F.interpolate(flow_cpu, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
        info_down = F.interpolate(info_cpu, scale_factor=0.5 ** args.scale, mode='area')
        
        flow_down = flow_down.to(flow.device)
        info_down = info_down.to(info.device)
    else:
        flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    
    return flow_down, info_down

@torch.no_grad()
def demo_data(path, args, model, image1, image2):
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    flow, info = calc_flow(args, model, image1, image2)
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}flow.jpg", flow_vis)
    heatmap = get_heatmap(info, args)
    vis_heatmap(f"{path}heatmap.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())

@torch.no_grad()
def demo_custom(model, args, device=torch.device('cuda')):
    image1 = cv2.imread("./custom/image1.jpg")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread("./custom/image2.jpg")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    demo_data('./custom/', args, model, image1, image2)

@torch.no_grad()
def demo_video(model, args, video_path, output_dir, device=torch.device('cuda')):
    """
    对AVI视频进行光流分析
    
    :param model: RAFT模型
    :param args: 参数
    :param video_path: 输入视频路径
    :param output_dir: 输出目录
    :param device: 设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    flow_video_path = os.path.join(output_dir, 'flow_video.avi')
    heatmap_video_path = os.path.join(output_dir, 'heatmap_video.avi')
    
    flow_writer = cv2.VideoWriter(flow_video_path, fourcc, fps, (width, height))
    heatmap_writer = cv2.VideoWriter(heatmap_video_path, fourcc, fps, (width, height + 50))
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
    frame_count = 0
    
    while True:
        # 读取下一帧
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # 转换颜色空间
        prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor
        prev_tensor = torch.tensor(prev_rgb, dtype=torch.float32).permute(2, 0, 1)
        curr_tensor = torch.tensor(curr_rgb, dtype=torch.float32).permute(2, 0, 1)
        
        # 添加batch维度并移动到设备
        prev_tensor = prev_tensor[None].to(device)
        curr_tensor = curr_tensor[None].to(device)
        
        try:
            # 计算光流
            flow, info = calc_flow(args, model, prev_tensor, curr_tensor)
            
            # 可视化光流
            flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
            
            # 可视化热力图
            heatmap = get_heatmap(info, args)
            heatmap_np = heatmap[0].permute(1, 2, 0).cpu().numpy()
            heatmap_np = heatmap_np[:, :, 0]
            heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
            heatmap_np = (heatmap_np * 255).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
            
            # 创建热力图叠加
            prev_rgb_uint8 = prev_rgb.astype(np.uint8)
            overlay = prev_rgb_uint8 * 0.3 + colored_heatmap * 0.7
            overlay = overlay.astype(np.uint8)
            
            # 添加颜色条到热力图
            color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)
            heatmap_with_bar = add_color_bar_to_image(overlay, color_bar, 'vertical')
            heatmap_with_bar_bgr = cv2.cvtColor(heatmap_with_bar, cv2.COLOR_RGB2BGR)
            
            # 写入视频帧
            flow_writer.write(flow_vis)
            heatmap_writer.write(heatmap_with_bar_bgr)
            
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            # 跳过这一帧，继续处理下一帧
            prev_frame = curr_frame
            frame_count += 1
            continue
        
        # 每处理10帧输出一次进度
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
        
        # 更新前一帧
        prev_frame = curr_frame
    
    # 释放资源
    cap.release()
    flow_writer.release()
    heatmap_writer.release()
    
    print(f"Video processing completed!")
    print(f"Flow video saved to: {flow_video_path}")
    print(f"Heatmap video saved to: {heatmap_video_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--path', help='checkpoint path', type=str, default=None)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    parser.add_argument('--mode', help='demo mode: custom or video', type=str, default='video', choices=['custom', 'video'])
    parser.add_argument('--video', help='input video path for video mode', type=str, default=None)
    parser.add_argument('--output', help='output directory for video mode', type=str, default='./video_output/')
    
    args = parse_args(parser)
    
    # 打印CUDA信息
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    if args.path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    
    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFT.from_pretrained(args.url, args=args)
        
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("Using CPU for inference")
    
    model = model.to(device)
    model.eval()
    
    if args.mode == 'custom':
        demo_custom(model, args, device=device)
    elif args.mode == 'video':
        if args.video is None:
            raise ValueError("--video must be provided for video mode")
        demo_video(model, args, args.video, args.output, device=device)

if __name__ == '__main__':
    # 设置环境变量来帮助调试
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
