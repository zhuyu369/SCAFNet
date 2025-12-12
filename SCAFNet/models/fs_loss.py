import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_msssim
from models.weight_block import weight_block


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image1, image2):
        img1_grad = self.sobelconv(image1)
        img2_grad = self.sobelconv(image2)
        loss_grad = F.l1_loss(img2_grad, img1_grad)
        return loss_grad


def sliced_wasserstein_distance(x, y, num_projections=100):
    """
    Args:
        x: Tensor of shape (batch_size, channels, height, width)
        y: Tensor of shape (batch_size, channels, height, width)
        num_projections: Number of random projections

    Returns:
        Sliced Wasserstein distance between x and y
    """
    assert x.shape == y.shape, "Input tensors must have the same shape"

    # Flatten the input tensors
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)

    # Generate random projections
    projections = torch.randn(x.size(1), num_projections, device=x.device)

    # Normalize projections
    projections /= torch.norm(projections, dim=0, keepdim=True)

    # Project the data
    x_projections = torch.matmul(x, projections)
    y_projections = torch.matmul(y, projections)

    # Sort the projections
    x_projections = torch.sort(x_projections, dim=0).values
    y_projections = torch.sort(y_projections, dim=0).values

    # Compute the Wasserstein distance in 1D for each projection
    swd = torch.mean(torch.abs(x_projections - y_projections))

    return swd


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image1, image2, generate_img):
        wb = weight_block(image1, image2)
        # mri_loss = torch.mean(torch.square(generate_img - image1))
        # pet_loss = torch.mean(torch.square(generate_img - image2))
        # loss_in = mri_loss + pet_loss

        # # 强度损失1
        # x_in_max = torch.max(image1, image2)
        # loss_in = F.l1_loss(generate_img, x_in_max)
        # 强度损失2
        loss_in = wb[0] * F.l1_loss(generate_img, image1) + \
            wb[1] * F.l1_loss(generate_img, image2)
        # Gradient 梯度损失
        img1_grad = self.sobelconv(image1)
        img2_grad = self.sobelconv(image2)
        fusion_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(img1_grad, img2_grad)
        loss_grad = F.l1_loss(fusion_img_grad, x_grad_joint)
        # SSIM损失，结构相似性损失
        ssim_loss = pytorch_msssim.msssim
        ssim_loss_temp1 = ssim_loss(generate_img, image1, normalize=True)
        ssim_loss_temp2 = ssim_loss(generate_img, image2, normalize=True)
        loss_ssim = wb[0] * (1 - ssim_loss_temp1) + \
            wb[0] * (1 - ssim_loss_temp2)
        # wd_loss_temp1 = sliced_wasserstein_distance(image1, generate_img)
        # wd_loss_temp2 = sliced_wasserstein_distance(image2, generate_img)
        # loss_wd = wb[0]*wd_loss_temp1+wb[1]*wd_loss_temp2

        return loss_in, loss_grad, loss_ssim
