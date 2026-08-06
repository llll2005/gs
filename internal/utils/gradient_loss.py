import torch

def compute_gradient_loss(pixels, colors, edge_threshold=4, rgb_boundary_threshold=0.01):
    """
    Compute gradient-aware loss with masking
    
    Args:
        pixels: Target image tensor [B, H, W, C]
        colors: Rendered image tensor [B, H, W, C] 
        edge_threshold: Threshold for edge detection relative to median gradient
        rgb_boundary_threshold: Threshold for RGB boundary detection
    """
    def image_gradient(image):
        # Compute image gradient using Scharr Filter
        c = image.shape[0]
        conv_y = torch.tensor(
            [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
        )
        conv_x = torch.tensor(
            [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
        )
        normalizer = 1.0 / torch.abs(conv_y).sum()
        p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
        img_grad_v = normalizer * torch.nn.functional.conv2d(
            p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
        )
        img_grad_h = normalizer * torch.nn.functional.conv2d(
            p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
        )
        return img_grad_v[0], img_grad_h[0]


    def image_gradient_mask(image, eps=0.01):
        # Compute image gradient mask
        c = image.shape[0]
        conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
        p_img = torch.abs(p_img) > eps
        img_grad_v = torch.nn.functional.conv2d(
            p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
        )
        img_grad_h = torch.nn.functional.conv2d(
            p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
        )

        return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


    # Process each batch item
    batch_losses = []
    for b in range(pixels.shape[0]):
        # Convert target image to grayscale [1, H, W]
        gray_img = pixels[b].permute(2, 0, 1).mean(dim=0, keepdim=True)
        
        # Compute gradients and masks
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        
        # Apply masks to gradients
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        
        # Compute gradient intensity
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)
        
        # Create edge mask based on median threshold
        median_img_grad_intensity = torch.median(img_grad_intensity)
        image_mask = (img_grad_intensity > median_img_grad_intensity * edge_threshold).float()
        
        # Create RGB boundary mask
        rgb_pixel_mask = (pixels[b].sum(dim=-1) > rgb_boundary_threshold).float()
        
        # Combine masks
        combined_mask = image_mask * rgb_pixel_mask

        # Compute masked L1 loss
        batch_loss = combined_mask * torch.abs(colors[b] - pixels[b]).mean(dim=-1)
        batch_losses.append(batch_loss.sum() / (combined_mask.sum() + 1e-8))

    # Average losses across batch
    return torch.stack(batch_losses).mean()