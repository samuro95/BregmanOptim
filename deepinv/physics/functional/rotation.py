import torch
import torch.fft

def rotate_image_via_shear(image: torch.Tensor, angle_deg: torch.Tensor, center=None):
    # Convert angle to radians
    angle = torch.deg2rad(angle_deg)
    N0, N1 = image.shape[-2], image.shape[-1]
    if center is None:
        center = (N0//2, N1//2)
  
    mask_angles = (angle > torch.pi / 2.0) & (angle <=  3 * torch.pi / 2)

    angle[angle > 3 * torch.pi / 2] -= 2 * torch.pi 
    
    transformed_image = torch.zeros_like(image).expand(mask_angles.shape[0], -1, -1, -1).clone()
    expanded_image = image.clone().expand(mask_angles.shape[0], -1, -1, -1).clone()
    transformed_image[~mask_angles] = expanded_image[~mask_angles]
    transformed_image[mask_angles] = torch.rot90(expanded_image[mask_angles], k=-2, dims=(-2, -1))
    
    angle[mask_angles] -= torch.pi

    tant2 = - torch.tan(-angle/ 2)
    st = torch.sin(-angle)

    def shearx(image, shear):
        fft1 = torch.fft.fft2(image, dim=(-1))
        freq_1 = torch.fft.fftfreq(N1, d=1.0, device=image.device)
        freq_0 = shear[:, None] * (torch.arange(N0, device=image.device) - center[0])[None]
        phase_shift = torch.exp(-2j * torch.pi * freq_0[..., None] * freq_1[None, None, :])
        image_shear = fft1 * phase_shift[:, None]
        return torch.abs(torch.fft.ifft2(image_shear, dim=(-1)))
    
    def sheary(image, shear):
        fft0 = torch.fft.fft2(image, dim=(-2))
        freq_0 = torch.fft.fftfreq(N0, d=1.0, device=image.device)
        freq_1 = shear[:, None] * (torch.arange(N1, device=image.device) - center[1])[None]
        phase_shift = torch.exp(-2j * torch.pi * freq_0[None, :, None] * freq_1[:, None, :])
        image_shear = fft0 * phase_shift[:, None]
        return torch.abs(torch.fft.ifft2(image_shear, dim=(-2)))
        
    rot = shearx(sheary(shearx(transformed_image, tant2), st), tant2)
    return rot 