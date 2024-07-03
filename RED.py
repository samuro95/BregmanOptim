import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import RED
from deepinv.optim.optimizers import optim_builder
from deepinv.training import test
from torchvision import transforms
from deepinv.utils.parameters import get_GSPnP_params
from deepinv.utils.demo import load_dataset, load_degradation

# %%
# Setup paths for data loading and results.
# --------------------------------------------------------
#

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"

# Set the global random seed from pytorch to ensure
# the reproducibility of the example.
torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# --------------------------------------------------------------------------------
# In this example, we use the Set3C dataset and a motion blur kernel from
# `Levin et al. (2009) <https://ieeexplore.ieee.org/abstract/document/5206815/>`_.
# Set up the variable to fetch dataset and operators.

method = "DPIR"
dataset_name = "set3c"
img_size = 256 if torch.cuda.is_available() else 32
img_size = 64
val_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)

# Generate a motion blur operator.
kernel_index = 1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
kernel_torch = load_degradation("Levin09.npy", DEG_DIR / "kernels", index=kernel_index)
kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
    0
)  # add batch and channel dimensions
dataset = load_dataset(dataset_name, ORIGINAL_DATA_DIR, transform=val_transform)

# Use parallel dataloader if using a GPU to fasten training, otherwise, as all computes are on CPU, use synchronous dataloading.
num_workers = 4 if torch.cuda.is_available() else 0

noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
n_channels = 3  # 3 for color images, 1 for gray-scale images
p = dinv.physics.BlurFFT(
    img_size=(n_channels, img_size, img_size),
    filter=kernel_torch,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)

# Use parallel dataloader if using a GPU to fasten training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0

n_images_max = 1  # Maximal number of images to restore from the input dataset
# Generate a dataset in a HDF5 folder in "{dir}/dinv_dataset0.h5'" and load it.
operation = "deblur"
measurement_dir = DATA_DIR / dataset_name / operation
dinv_dataset_path = dinv.datasets.generate_dataset(
    train_dataset=dataset,
    test_dataset=None,
    physics=p,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
)

batch_size = 1  # batch size for testing. As the number of iterations is fixed, we can use batch_size > 1
# and restore multiple images in parallel.
dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=True)

# %%
# Setup the PnP algorithm. This involves in particular the definition of a custom prior class.
# ------------------------------------------------------------------------------------------------------
# We use the proximal gradient algorithm to solve the super-resolution problem with GSPnP.

# Parameters of the algorithm to solve the inverse problem
early_stop = True  # Stop algorithm when convergence criteria is reached
crit_conv = "residual"  # Convergence is reached when the difference of cost function between consecutive iterates is
# smaller than thres_conv
thres_conv = 1e-5
backtracking = True
batch_size = 1  # batch size for evaluation is necessarily 1 for early stopping and backtracking to work.

# load specific parameters for GSPnP
lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params(operation, noise_level_img)

params_algo = {
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "lambda": lamb,
}

# Select the data fidelity term
data_fidelity = L2()


# The GSPnP prior corresponds to a RED prior with an explicit `g`.
# We thus write a class that inherits from RED for this custom prior.
class ExplicitRED(dinv.optim.Prior):
    def __init__(self, denoiser, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.denoiser = denoiser

    def g(self, x, sigma_denoiser, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return torch.sum((x - self.denoiser(x,sigma_denoiser))*x, dim=(1, 2, 3))


# Specify the Denoising prior
prior_1 = RED(
    denoiser=dinv.models.DRUNet(pretrained='download', train=False, device=device)
)

prior_2 = ExplicitRED(
    denoiser=dinv.models.DRUNet(pretrained='download', train=False, device=device)
)

# prior_1 = RED(
#     denoiser=dinv.models.DRUNetConditional(pretrained='../rhomonotone/ckpts/state_dict_ckp_210.pth.tar', train=False, device=device)
# )

# prior_2 = ExplicitRED(
#     denoiser=dinv.models.DRUNetConditional(pretrained='../rhomonotone/ckpts/state_dict_ckp_210.pth.tar', train=False, device=device)
# )


# we want to output the intermediate PGD update to finish with a denoising step.
def custom_output(X):
    return X["est"][1]

dataloader = DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

# params_algo['stepsize'] = 1.

# n_iter = 100
# x = next(iter(dataloader))[0].to(device)
# y = p(x)
# x1 = x2 = y
# x1_list = []
# x2_list = []
# for i in range(n_iter):
#     x1 = x1 - params_algo["stepsize"] * params_algo["lambda"] * prior_1.grad(x1, params_algo["g_param"])
#     x1 = data_fidelity.prox(x1, y, p, gamma = params_algo["stepsize"])
#     x2 = x2 - params_algo["stepsize"] * params_algo["lambda"] * prior_2.grad(x2, params_algo["g_param"])
#     x2 = data_fidelity.prox(x2, y, p, gamma = params_algo["stepsize"])
#     # out_1 = x1 - params_algo["stepsize"] * params_algo["lambda"] * prior_1.grad(x1, params_algo["g_param"])
#     # out_2 = x2 - params_algo["stepsize"] * params_algo["lambda"] * prior_2.grad(x2, params_algo["g_param"])
#     x1_list.append(x1.detach().cpu())
#     x2_list.append(x2.detach().cpu())
# diff_list = [torch.norm(x1 - x2) for x1, x2 in zip(x1_list, x2_list)]
# x1_conv = [torch.norm(x1_list[i+1] - x1_list[i]) for i in range(len(x1_list)-1)]
# x2_conv = [torch.norm(x2_list[i+1] - x2_list[i]) for i in range(len(x2_list)-1)]
# import matplotlib.pyplot as plt
# plt.figure(0)
# plt.plot(diff_list)
# plt.figure(1)
# plt.semilogy(x1_conv, label = 'RED')
# plt.semilogy(x2_conv, label = 'symmetric RED')
# plt.legend()
# plt.show()


# # instantiate the algorithm class to solve the IP problem.
# model_1 = optim_builder(
#     iteration="GD",
#     prior=prior_1,
#     g_first=True,
#     data_fidelity=data_fidelity,
#     params_algo=params_algo,
#     early_stop=early_stop,
#     max_iter=max_iter,
#     crit_conv=crit_conv,
#     thres_conv=thres_conv,
#     backtracking=backtracking,
#     get_output=custom_output,
#     verbose=True,
# )

# model_2 = optim_builder(
#     iteration="PGD",
#     prior=prior_2,
#     g_first=True,
#     data_fidelity=data_fidelity,
#     params_algo=params_algo,
#     early_stop=early_stop,
#     max_iter=max_iter,
#     crit_conv=crit_conv,
#     thres_conv=thres_conv,
#     backtracking=backtracking,
#     get_output=custom_output,
#     verbose=True,
# )

# # %%
# # Evaluate the model on the problem.
# # ----------------------------------------------------
# # We evaluate the PnP algorithm on the test dataset, compute the PSNR metrics and plot reconstruction results.

save_folder = RESULTS_DIR / method / operation / dataset_name
wandb_vis = False  # plot curves and images in Weight&Bias.
plot_metrics = True  # plot metrics. Metrics are saved in save_folder.
plot_images = True  # plot images. Images are saved in save_folder.


# with torch.no_grad():
#     test(
#         model=model_2,
#         test_dataloader=dataloader,
#         physics=p,
#         device=device,
#         plot_images=plot_images,
#         save_folder=RESULTS_DIR / method / operation / dataset_name,
#         plot_metrics=plot_metrics,
#         verbose=True,
#         wandb_vis=wandb_vis,
#         plot_only_first_batch=False,  # By default only the first batch is plotted.
#     )

# with torch.no_grad():
#     test(
#         model=model_1,
#         test_dataloader=dataloader,
#         physics=p,
#         device=device,
#         plot_images=plot_images,
#         save_folder=RESULTS_DIR / method / operation / dataset_name,
#         plot_metrics=plot_metrics,
#         verbose=True,
#         wandb_vis=wandb_vis,
#         plot_only_first_batch=False,  # By default only the first batch is plotted.
#     )
thres_conv = 1e-5
import matplotlib.pyplot as plt



# params_algo['lambda'] = 1.
# params_algo['stepsize'] = 0.1

# n_images_max = 3
# max_iter = 500
# for k in range(n_images_max):
#     y = p(next(iter(dataloader))[0].to(device))
#     x = y
#     norm_tab = []
#     T_tab = []
#     iter_tab = []
#     for i in range(max_iter):
#         x_old = x
#         grad = params_algo["lambda"]*prior_1.grad(x_old, params_algo["g_param"])
#         T_tab.append(grad)
#         iter_tab.append(x_old)
#         x = x_old - params_algo["stepsize"] * grad
#         norm = torch.norm(x - x_old).item()
#         norm_tab.append(norm)
#         if norm < thres_conv or norm > 1e5:
#             break
#     T = params_algo["lambda"]*prior_1.grad(x, params_algo["g_param"])
#     MVI_tab = [(torch.sum((T_tab[i] - T)*(iter_tab[i] - x)) / torch.norm(T_tab[i] - T)**2).item() for i in range(len(T_tab))]
#     print('Image', k, 'min', min(MVI_tab))
#     plt.plot(MVI_tab)
#     # plt.semilogy(norm_tab)
# plt.show()


class GSPnP(RED):
    r"""
    Gradient-Step Denoiser prior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.denoiser.potential(x, *args, **kwargs)


# Specify the Denoising prior
# prior_1 = GSPnP(
#     denoiser=dinv.models.DRUNet(pretrained="download", train=False).to(device)
# )


print(params_algo['g_param'])
params_algo['stepsize'] = 1.
params_algo['g_param'] = 0.05

def Jvp(y, x, u):
    w = torch.ones_like(y, requires_grad=True)
    return torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=True)[0]

def vJp(y, x, u):
    return torch.autograd.grad(y, x, u, retain_graph=True, create_graph=True)[0]   

with torch.no_grad():
    n_images_max = 1
    max_iter = 10000
    for k in range(n_images_max):
        print('Image', k)
        y = next(iter(dataloader))[0].to(device)
        x = y
        norm_tab = []
        T_tab = []
        iter_tab = []
        J_norm = []
        for i in range(max_iter):
            x_old = x
            grad = params_algo["lambda"]*prior_1.grad(x_old, params_algo["g_param"])
            grad.detach()
            x_old.detach()
            T_tab.append(grad)
            iter_tab.append(x_old)
            x = x_old - params_algo["stepsize"] * grad
            x = torch.clip(x, 0, 1)
            norm = torch.norm(x - x_old).item()
            norm_tab.append(norm)
            if norm < thres_conv:
                print('Image', k, 'converged at iteration', i)
                break
            # f = lambda z : prior_1.denoiser(z, params_algo["g_param"])
            # jvp = torch.autograd.functional.jvp(f, x, v = x)[1]
            # vjp = torch.autograd.functional.vjp(f, x, v = x)[1]
            # J_norm.append(torch.norm(jvp-vjp))
        T = prior_1.grad(x, params_algo["g_param"])
        MVI_tab = [(torch.sum((T_tab[i] - T)*(iter_tab[i] - x)) / torch.norm(T_tab[i] - T)**2).item() for i in range(len(T_tab))]
        mono_tab = [(torch.sum((T_tab[i] - T)*(iter_tab[i] - x)) / torch.norm(iter_tab[i] - x)**2).item() for i in range(len(T_tab))]
        Lip_tab = [norm_tab[i+1] / norm_tab[i] for i in range(len(norm_tab)-1)]
        # move_tab = [(torch.norm(T_tab[i] - T) / torch.norm(iter_tab[i] - x)).item() for i in range(len(T_tab))]
        # print('Image', k, 'min', min(MVI_tab))
        plt.figure(0)
        plt.plot(MVI_tab)
        plt.figure(1)
        plt.plot(mono_tab)
        plt.figure(3)
        plt.semilogy(norm_tab)
        plt.figure(4)
        plt.plot(Lip_tab)
        dinv.utils.plot([x])
    plt.show()






