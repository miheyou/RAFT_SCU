import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import math
import cv2
import os
import argparse
import sys
sys.path.append('core')

# 处理导入和兼容性
try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import error: {e}")
    sys.exit(1)


# 检查必要的模块
try:
    import h5py
    print("✓ h5py imported successfully")
except ImportError:
    print("✗ h5py not available, installing...")
    os.system("pip install h5py")

try:
    from config.parser import parse_args
    from raft import RAFT
    from utils.flow_viz import flow_to_image
    from utils.utils import load_ckpt
    print("✓ RAFT modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def setup_device(device_type='auto'):
    """设置计算设备"""
    if device_type == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            try:
                # 测试CUDA
                test_tensor = torch.randn(100, 100).cuda()
                _ = test_tensor * test_tensor
                torch.cuda.synchronize()

                print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
                print(f"  CUDA version: {torch.version.cuda}")
                print(
                    f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

                # 启用CUDA优化
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True

            except Exception as e:
                print(f"✗ CUDA test failed: {e}")
                device = torch.device('cpu')
                print("ℹ Falling back to CPU")
        else:
            device = torch.device('cpu')
            print("ℹ CUDA not available, using CPU")

    elif device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU as requested")
    else:
        device = torch.device('cpu')
        print("ℹ Using CPU as requested")

    return device


def create_color_bar(height, width, color_map=cv2.COLORMAP_JET):
    """创建颜色条"""
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)
    color_bar = cv2.applyColorMap(gradient, color_map)
    return color_bar


def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """添加颜色条到图像"""
    if orientation == 'vertical':
        return np.vstack([image, color_bar])
    else:
        return np.hstack([image, color_bar])


def get_heatmap(info, args):
    """计算热力图"""
    try:
        raw_b = info[:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = info[:, :2].softmax(dim=1)
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
        heatmap = (log_b * weight).sum(dim=1, keepdim=True)
        return heatmap
    except Exception as e:
        print(f"⚠ Heatmap calculation warning: {e}")
        # 备用热力图
        return info[:, :1]


def forward_flow(args, model, image1, image2):
    """前向计算光流"""
    try:
        with torch.no_grad():
            output = model(image1, image2, iters=args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final
    except Exception as e:
        print(f"✗ Forward flow error: {e}")
        # 返回零光流作为备用
        batch_size, _, H, W = image1.shape
        flow = torch.zeros(batch_size, 2, H, W, device=image1.device)
        info = torch.zeros(batch_size, 4, H, W, device=image1.device)
        return flow, info


def calc_flow(args, model, image1, image2):
    """计算光流"""
    try:
        # 直接在设备上处理
        img1 = F.interpolate(image1, scale_factor=2 **
                             args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 **
                             args.scale, mode='bilinear', align_corners=False)

        flow, info = forward_flow(args, model, img1, img2)

        # 下采样
        flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale,
                                  mode='bilinear', align_corners=False) * (0.5 ** args.scale)
        info_down = F.interpolate(
            info, scale_factor=0.5 ** args.scale, mode='area')

        return flow_down, info_down

    except Exception as e:
        print(f"✗ Flow calculation error: {e}")
        batch_size, _, H, W = image1.shape
        flow = torch.zeros(batch_size, 2, H, W, device=image1.device)
        info = torch.zeros(batch_size, 4, H, W, device=image1.device)
        return flow, info


@torch.no_grad()
def process_video(model, args, video_path, output_dir, device):
    """视频处理主函数"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return False

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"📹 Video info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    print(f"💻 Using device: {device}")

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    flow_video_path = os.path.join(output_dir, 'flow_video.avi')
    heatmap_video_path = os.path.join(output_dir, 'heatmap_video.avi')

    flow_writer = cv2.VideoWriter(
        flow_video_path, fourcc, fps, (width, height))
    heatmap_writer = cv2.VideoWriter(
        heatmap_video_path, fourcc, fps, (width, height + 50))

    if not flow_writer.isOpened() or not heatmap_writer.isOpened():
        print("✗ Cannot create video writers")
        return False

    # 清空GPU缓存（如果可用）
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("✗ Cannot read first frame")
        return False

    # 预处理第一帧
    prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    prev_tensor = torch.from_numpy(prev_rgb).float().permute(
        2, 0, 1).unsqueeze(0).to(device)

    frame_count = 0
    processed_count = 0
    total_time = 0

    print("🚀 Starting video processing...")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        try:
            # 预处理当前帧
            curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            curr_tensor = torch.from_numpy(curr_rgb).float().permute(
                2, 0, 1).unsqueeze(0).to(device)

            # 计算光流
            flow, info = calc_flow(args, model, prev_tensor, curr_tensor)

            # 可视化光流
            flow_np = flow[0].permute(
                1, 2, 0).cpu().numpy()  # shape: [H, W, 2]

            # 计算光流幅值: sqrt(flow_x^2 + flow_y^2)
            flow_mag = np.sqrt(flow_np[:, :, 0]**2 + flow_np[:, :, 1]**2)

            # 设置一个阈值，例如 0.5 （可根据实际情况调整，一般 0.3 ~ 1.0）
            flow_threshold = 2.0

            # 构造一个 mask，只保留大于阈值的部分
            mask = flow_mag > flow_threshold

            # 将小于阈值的光流置零（或可设为 nan，但需后续处理）
            filtered_flow = flow_np.copy()
            filtered_flow[~mask] = 0  # 抑制小幅光流

            # 可视化过滤后的光流
            flow_vis = flow_to_image(filtered_flow, convert_to_bgr=True)

            # 计算热力图
            heatmap = get_heatmap(info, args)
            heatmap_np = heatmap[0].permute(1, 2, 0).cpu().numpy()[:, :, 0]
            heatmap_np = (heatmap_np - heatmap_np.min()) / \
                (heatmap_np.max() - heatmap_np.min() + 1e-8)
            heatmap_np = (heatmap_np * 255).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

            # 创建热力图叠加
            overlay = prev_rgb.astype(np.uint8) * 0.3 + colored_heatmap * 0.7
            overlay = overlay.astype(np.uint8)

            # 添加颜色条
            color_bar = create_color_bar(50, width)
            heatmap_with_bar = add_color_bar_to_image(overlay, color_bar)
            heatmap_with_bar_bgr = cv2.cvtColor(
                heatmap_with_bar, cv2.COLOR_RGB2BGR)

            # 写入视频
            flow_writer.write(flow_vis)
            heatmap_writer.write(heatmap_with_bar_bgr)

            # 更新状态
            processing_time = time.time() - start_time
            total_time += processing_time
            processed_count += 1

            # 更新前一帧
            prev_frame = curr_frame
            prev_rgb = curr_rgb.copy()
            prev_tensor = curr_tensor

        except Exception as e:
            print(f"⚠ Frame {frame_count} error: {e}")

        # 进度显示
        frame_count += 1
        if frame_count % 10 == 0:
            avg_time = total_time / processed_count if processed_count > 0 else 0
            fps_est = 1.0 / avg_time if avg_time > 0 else 0

            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1024**3
                mem_str = f"GPU: {memory_used:.1f}GB"
            else:
                mem_str = "CPU"

            print(f"📊 Frame {frame_count:4d}/{total_frames} | "
                  f"Success: {processed_count:4d} | "
                  f"Time: {avg_time:.3f}s | "
                  f"FPS: {fps_est:.1f} | "
                  f"{mem_str}")

    # 清理资源
    cap.release()
    flow_writer.release()
    heatmap_writer.release()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 输出总结
    print(f"\n✅ Processing completed!")
    print(f"   Processed frames: {processed_count}/{total_frames}")
    if processed_count > 0:
        print(f"   Average time per frame: {total_time/processed_count:.3f}s")
        print(f"   Average FPS: {1.0/(total_time/processed_count):.1f}")
    print(f"   Flow video: {flow_video_path}")
    print(f"   Heatmap video: {heatmap_video_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description='RAFT Video Optical Flow')
    parser.add_argument('--cfg', required=True, help='Config file')
    parser.add_argument('--path', help='Checkpoint path')
    parser.add_argument('--url', help='Checkpoint URL')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', default='./output/',
                        help='Output directory')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--scale', type=int, default=1,
                        help='Scale factor for multi-scale (e.g., 1 or 2)')
    parser.add_argument('--iters', type=int, default=12,
                        help='Number of iterations for RAFT')
    parser.add_argument('--var_max', type=float, default=10.0,
                        help='Max value for heatmap var 1')
    parser.add_argument('--var_min', type=float, default=-
                        10.0, help='Min value for heatmap var 2')

    args = parse_args(parser)

    print("🚀 RAFT Video Optical Flow")
    print("=" * 50)

    # 设置设备
    device = setup_device(args.device)

    # 检查参数
    if args.path is None and args.url is None:
        print("✗ Either --path or --url must be provided")
        return

    # 加载模型
    try:
        if args.path is not None:
            print(f"📁 Loading model from: {args.path}")
            model = RAFT(args)
            load_ckpt(model, args.path)
        else:
            print(f"🌐 Loading model from URL: {args.url}")
            model = RAFT.from_pretrained(args.url, args=args)

        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return

    # 处理视频
    success = process_video(model, args, args.video, args.output, device)

    if success:
        print("\n🎉 All done!")
    else:
        print("\n❌ Processing failed!")


if __name__ == '__main__':
    main()
