import torch
from deepinv.physics.generator import PhysicsGenerator


class SigmaGenerator(PhysicsGenerator):
    r"""
    Generator for the noise level :math:`\sigma` in the Gaussian noise model.

    The noise level is sampled uniformly from the interval :math:`[\text{sigma_min}, \text{sigma_max}]`.

    :param float sigma_min: minimum noise level
    :param float sigma_max: maximum noise level
    :param str device: device where the tensor is stored

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import SigmaGenerator
    >>> generator = SigmaGenerator()
    >>> _ = torch.manual_seed(0)
    >>> sigma_dict = generator.step() # dict_keys(['sigma'])
    >>> print(sigma_dict['sigma'])
    tensor([0.2532])

    """

    def __init__(self, sigma_min=0.01, sigma_max=0.5, log_uniform=False, device: str = "cpu"):
        super().__init__(shape=(1,), device=device)
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_uniform = log_uniform

    def step(self, batch_size=1, **kwargs):
        r"""
        Generates a batch of noise levels.

        :param int batch_size: batch size

        :return: dictionary with key **'sigma'**: tensor of size (batch_size,).
        :rtype: dict

        """
        if self.log_uniform:
            log_sigma_min = torch.log(torch.tensor([self.sigma_min]))
            log_sigma_max = torch.log(torch.tensor([self.sigma_max]))
            
            log_sigma = (
                torch.rand(batch_size, device=self.device)
                * (log_sigma_max - log_sigma_min)
                + log_sigma_min
            )
            
            sigma = torch.exp(log_sigma)
        else:
            sigma = (
                torch.rand(batch_size, device=self.device)
                * (self.sigma_max - self.sigma_min)
                + self.sigma_min
            )
        return {"sigma": sigma}


# # if __name__ == "__main__":

# #%%
# import deepinv as dinv
# from deepinv.utils.demo import load_url_image, get_image_url
# from pathlib import Path
# import imageio.v2 as io
# import matplotlib.pyplot as plt
# from display_utils import show_images
# import numpy as np

# device = 'cuda'
# dtype = torch.float32
# img_size = (173, 125)

# # url = get_image_url("barbara.jpeg")
# # x = load_url_image(
# #     url, grayscale=True, device=device, dtype=dtype, img_size=img_size
# # )

# folder_name = Path("/run/user/1006/gvfs/smb-share:server=cbi-filer-01,share=equipes/CBI097/databases/cellpose_data/train/")
# file_img = folder_name / "221_img.png"
# file_bnd = folder_name / "221_boundaries.png"
# file_lbl = folder_name / "001_labels.png"
# x = torch.from_numpy(io.imread(file_img)[:, :, 1]).to(dtype).to(device)[None]
# x /= x.max()
    
# gain_min=0.0001
# gain_max=0.1

# batch_size=5
# x = x.expand(batch_size, -1, -1)

# #sigma_generator = SigmaGenerator(sigma_min=gain_min, sigma_max=gain_max, device=device)#log_uniform=True)
# #gain = sigma_generator.step(batch_size)['sigma']

# #gain = torch.linspace(gain_min, gain_max, batch_size, device=device)
# gain = torch.logspace(np.log10(gain_min), np.log10(gain_max), batch_size, device=device)

# #%%
# y_poisson = torch.poisson(x / gain[:, None, None]) * gain[:, None, None]

# #for i in range(batch_size):
# #    show_images(y[i, None, ...])

# #_ = plt.imshow(x[0].cpu()); plt.show(); plt.imshow(y[0].cpu()); plt.show()



# ## Gaussian

# sigma_min=0.01
# sigma_max=0.05

# sigma = torch.logspace(np.log10(sigma_min), np.log10(sigma_max), batch_size, device=device)


# #%%

# y_pg = torch.zeros(batch_size, batch_size, 1, x.shape[-2], x.shape[-1])

# for i in range(batch_size):
#     y = torch.poisson(x[i] / gain[i, None, None]) * gain[i, None, None]
#     for j in range(batch_size):
#         y_pg[i, j, 0] = y_poisson[i] + torch.randn_like(x[i]) * sigma[j]
#         y_pg[i, j, 0] /= y_pg[i, j, 0].max()


# # for i in range(batch_size):
# #     y_gaussian = x + torch.randn_like(x[i]) * sigma[i]
# #     show_images([y_poisson[i:i+1, None], y_gaussian[i:i+1, None, ...]], colorbar=True)
    
    
# #     show_images([y_poisson[i:i+1, None], y_gaussian[i:i+1, None, ...]], colorbar=True)

# #     show_images([y_poisson[i:i+1, None], y_gaussian[i:i+1, None, ...]-x[i:i+1, None]], colorbar=True)




# show_images(y_pg.reshape(-1, 1, 1, x.shape[-2], x.shape[-1])[:, 0], ncols=5, savename='/home/fsarron/Videos/noise.pdf') #, vmin=y_pg.min(), vmax=y_pg.max())




