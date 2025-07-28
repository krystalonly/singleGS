import cv2
import numpy as np
import lpips
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from torchvision import transforms
from PIL import Image


def compute_psnr_and_ssim(reference_video_path, generated_video_path):
    reference_cap = cv2.VideoCapture(reference_video_path)
    generated_cap = cv2.VideoCapture(generated_video_path)

    psnr_values = []
    ssim_values = []

    while True:
        ret_ref, frame_ref = reference_cap.read()
        ret_gen, frame_gen = generated_cap.read()

        if not ret_ref or not ret_gen:
            break  # 视频读取结束

        # 转换为灰度图像
        gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
        gray_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2GRAY)

        # 计算 PSNR 和 SSIM
        psnr_value = psnr(gray_ref, gray_gen)
        ssim_value = ssim(gray_ref, gray_gen)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    reference_cap.release()
    generated_cap.release()

    return avg_psnr, avg_ssim


def compute_lpips(reference_video_path, generated_video_path):
    lpips_model = lpips.LPIPS(net='alex')  # 加载LPIPS模型
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])  # 将图片转换为[-1, 1]范围

    lpips_values = []

    reference_cap = cv2.VideoCapture(reference_video_path)
    generated_cap = cv2.VideoCapture(generated_video_path)

    while True:
        ret_ref, frame_ref = reference_cap.read()
        ret_gen, frame_gen = generated_cap.read()

        if not ret_ref or not ret_gen:
            break

        # 将图像转换为Tensor
        img1_tensor = transform(Image.fromarray(frame_ref)).unsqueeze(0)
        img2_tensor = transform(Image.fromarray(frame_gen)).unsqueeze(0)

        # 计算LPIPS
        lpips_value = lpips_model(img1_tensor, img2_tensor)
        lpips_values.append(lpips_value.item())

    avg_lpips = np.mean(lpips_values)

    reference_cap.release()
    generated_cap.release()

    return avg_lpips


# def compute_fid(reference_video_path, generated_video_path):
#     # 使用PyTorch FID库来计算FID
#     fid_value = fid_score.calculate_fid_given_paths([reference_video_path, generated_video_path], batch_size=1,
#                                                     device='cuda', dims=2048, num_workers=0)
#     return fid_value


def main(reference_video_path, generated_video_path):
    print("计算PSNR和SSIM...")
    avg_psnr, avg_ssim = compute_psnr_and_ssim(reference_video_path, generated_video_path)
    print(f"平均PSNR: {avg_psnr} dB")
    print(f"平均SSIM: {avg_ssim}")

    print("\n计算LPIPS...")
    avg_lpips = compute_lpips(reference_video_path, generated_video_path)
    print(f"平均LPIPS: {avg_lpips}")

    # print("\n计算FID...")
    # fid_value = compute_fid(reference_video_path, generated_video_path)
    # print(f"FID: {fid_value}")


if __name__ == "__main__":
    # 真实视频路径和生成的视频路径
    reference_video_path = "/media/sxm/Data/smh/exp/GaussianTalker/output/obama/test/ours_10000/gt/output_test_10000iter_gt.mov"
    generated_video_path = "/media/sxm/Data/smh/exp/GaussianTalker/output/obama/test/ours_10000/renders/output_test_10000iter_renders.mov"

    main(reference_video_path, generated_video_path)
