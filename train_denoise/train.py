import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import deepinv as dinv
from argparse import ArgumentParser
import json
from pathlib import Path
from models.unrolled_dual_MD import get_unrolled_architecture, MirrorLoss, NoLipLoss, FunctionalMetric, LambdaMetric, StepsizeMetric, SigmaMetric
from utils.distributed_setup import setup_distributed
from utils.dataloaders import get_drunet_dataset
from utils.utils import get_wandb_setup
from deepinv.physics.generator import GainGenerator, SigmaGenerator, MotionBlurGenerator, GaussianBlurGenerator, BernoulliSplittingMaskGenerator
from my_trainer import MyTrainer

torch.backends.cudnn.benchmark = True

with open("config/config.json") as json_file:
    config = json.load(json_file)

TRAIN_DATASET_PATH = config["TRAIN_DATASET_PATH"]
VAL_DATASET_PATH = None
WANDB_LOGS_PATH = config["WANDB_LOGS_PATH"]
PRETRAINED_PATH = config["PRETRAINED_PATH"]
OUT_DIR = Path(".")
CKPT_DIR = OUT_DIR / "ckpts"  # path to store the checkpoints
WANDB_PROJ_NAME = "learned_MD_denoising"  # Name of the wandb project

def load_denoising_data(
    patch_size,
    train_batch_size,
    val_batch_size,
    num_workers,
    device="cpu",
    split_val=0.9,
    distribute=False,
    max_num_images=1e6,
    noise_model = 'Gaussian',
    noise_level_min=0.,
    noise_level_max=0.2,
    degradation = None,
    psf_size = 31,
    downsampling_factor = 2,
    split_ratio = 0.5,
):
    """
    Load the training and validation datasets and create the corresponding dataloaders.

    :param torchvision.transforms train_transform: torchvision transform to be applied to the training data
    :param torchvision.transform val_transform: torchvision transform to be applied to the validation data
    :param int train_batch_size: training batch size
    :param int num_workers: number of workers
    :return: training and validation dataloaders
    """
    pin_memory = True if torch.cuda.is_available() else False
    dataset = get_drunet_dataset(
        patch_size,
        device=device,
        pth=TRAIN_DATASET_PATH,
        max_num_images=max_num_images,
    )
    # Calculate lengths for training and datasets (80% and 20%)
    train_size = int(split_val * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    if distribute and dist.is_initialized():
        # batch_size= train_batch_size * dist.get_world_size()
        train_batch_size = train_batch_size // dist.get_world_size()
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            dataset=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            dataset=num_workers,
            pin_memory=pin_memory,
            sampler=val_sampler,
        )

    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
        )

    img_size = (3, patch_size, patch_size)

    if noise_model.lower() == 'gaussian':
        noise = dinv.physics.GaussianNoise()
        noise_generator = SigmaGenerator(sigma_min=noise_level_min, sigma_max=noise_level_max, device=device)
    elif noise_model.lower() == 'poisson':
        noise = dinv.physics.PoissonNoise(clip_positive = True, normalize=True)
        noise_generator = GainGenerator(gain_min=noise_level_min, gain_max=noise_level_max, device=device)
    else:
        raise ValueError('noise model not available')

    if degradation.lower() == 'gaussian_blur' :
        blur_generator = GaussianBlurGenerator(psf_size=(psf_size, psf_size), sigma_min = 0.01, sigma_max= 4., num_channels=1, device=device) 
        generator = blur_generator + noise_generator
        physics = dinv.physics.BlurFFT(img_size = img_size, device=device, noise_model=noise)
    elif degradation.lower() == 'motion_blur' :
        blur_generator = MotionBlurGenerator((psf_size, psf_size), l=0.3, sigma=0.25, device=device)
        generator = blur_generator + noise_generator
        physics = dinv.physics.BlurFFT(img_size = img_size, device=device, noise_model=noise)
    elif degradation.lower() == 'inpainting' :
        mask_generator = BernoulliSplittingMaskGenerator(tensor_size = img_size, split_ratio = split_ratio, device = device)
        generator = mask_generator + noise_generator
        physics = dinv.physics.Inpainting(tensor_size = img_size, device=device, noise_model=noise)
    elif degradation.lower() == 'sr' :
        blur_generator = GaussianBlurGenerator(psf_size=(psf_size, psf_size), num_channels=1, device=device) 
        generator = blur_generator + noise_generator
        physics = dinv.physics.Downsampling(factor = downsampling_factor, img_size = img_size, device=device, noise_model=noise)
    else :
        generator = noise_generator
        physics = dinv.physics.DecomposablePhysics(device=device, noise_model=noise)

    val_dataloader = [val_dataloader]
    train_dataloaders = [train_dataloader]
    physics = [physics]
    generators = [generator]

    return train_dataloaders, val_dataloader, physics, generators


def train_model(
    ckpt_pretrained=None,
    test_only=False,
    n_layers=10,
    grayscale=False,
    gpu_num=1,
    model_name="dual_DDMD",
    prior_name="wavelet",
    denoiser_name="DRUNET",
    stepsize_init=1.0,
    lamb_init=1.0,
    sigma_denoiser_init = 0.03,
    wandb_resume_id=None,
    seed=0,
    wandb_vis=True,
    epochs=100,
    train_batch_size=16,
    val_batch_size=16,
    patch_size=64,
    lr=1e-4,
    distribute=False,
    num_workers=16 if torch.cuda.is_available() else 0,
    max_num_images=1e6,
    use_mirror_loss=False,
    data_fidelity="L2",
    noise_model="Gaussian",
    noise_level_min=0.,
    noise_level_max=0.2,
    strong_convexity_backward=0.5,
    strong_convexity_forward=0.1,
    strong_convexity_potential='L2',
    use_NoLip_loss=False,
    eps_jacobian_loss = 0.05, 
    jacobian_loss_weight = 1e-2, 
    max_iter_power_it=10, 
    tol_power_it=1e-3,
    degradation=None,
    args=None
):

    if distribute:
        device, global_rank = setup_distributed(seed)
    else:
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    operation = "denoising_MD"

    train_batch_size = train_batch_size * gpu_num
    train_dataloader, val_dataloader, physics, physics_generator = load_denoising_data(
        patch_size,
        train_batch_size,
        val_batch_size,
        num_workers,
        device=device,
        distribute=distribute,
        max_num_images=max_num_images,
        noise_model=noise_model,
        noise_level_min=noise_level_min,
        noise_level_max=noise_level_max,
        degradation=degradation
    )

    if not "dual" in model_name:
        use_mirror_loss = True
    else:
        use_dual_iterations = True

    model = get_unrolled_architecture(
        max_iter=n_layers,
        data_fidelity=data_fidelity,
        prior_name=prior_name,
        denoiser_name=denoiser_name,
        stepsize_init=stepsize_init,
        sigma_denoiser_init=sigma_denoiser_init,
        lamb_init=lamb_init,
        device=device,
        use_mirror_loss=use_mirror_loss,
        use_dual_iterations=use_dual_iterations,
        strong_convexity_backward = strong_convexity_backward,
        strong_convexity_forward = strong_convexity_forward,
        strong_convexity_potential = strong_convexity_potential
    )

    losses = [dinv.loss.SupLoss()]
    if use_mirror_loss:
        losses.append(MirrorLoss())
    if use_NoLip_loss:
        losses.append(NoLipLoss(eps_jacobian_loss = eps_jacobian_loss, jacobian_loss_weight = jacobian_loss_weight, max_iter_power_it=max_iter_power_it, tol_power_it=tol_power_it))

    metrics = [dinv.loss.metric.PSNR()]
    metrics.append(FunctionalMetric())
    metrics.append(LambdaMetric())
    metrics.append(StepsizeMetric())
    metrics.append(SigmaMetric())


    if dist.is_initialized() and distribute:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, gamma=0.1, step_size=int(9 * epochs / 10)
    )

    wandb_setup = get_wandb_setup(
        WANDB_LOGS_PATH, args, WANDB_PROJ_NAME, mode="online", wandb_resume_id=wandb_resume_id
    )

    if distribute:
        print("Start training on ", device)
        if global_rank == 0:
            show_progress_bar = True
            verbose = True
            plot_images = True
        else:
            show_progress_bar = False
            verbose = False
            plot_images = False
    else:
        show_progress_bar = True
        verbose = True
        plot_images = True

    print(
        "The model has ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters",
    )

    trainer = MyTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        epochs=epochs,
        scheduler=scheduler,
        losses=losses,
        physics=physics,
        physics_generator=physics_generator,
        optimizer=optimizer,
        device=device,
        save_path=str(CKPT_DIR / operation),
        verbose=verbose,
        wandb_vis=wandb_vis,
        wandb_setup=wandb_setup,
        plot_images=plot_images,
        eval_interval=1,
        ckp_interval=20,
        online_measurements=True,
        check_grad=True,
        ckpt_pretrained=ckpt_pretrained,
        freq_plot=1,
        show_progress_bar=show_progress_bar,
        display_losses_eval=True,
        metrics=metrics
    )
    
    trainer.test(val_dataloader)

    if not test_only:
        trainer.train()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--test_only", type=int, default=0)
    parser.add_argument("--grayscale", type=int, default=0)
    parser.add_argument("--ckpt_pretrained", type=str)
    parser.add_argument("--data_fidelity", type=str, default="KL")
    parser.add_argument("--noise_model", type=str, default="Poisson")
    parser.add_argument("--degradation", type=str)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="dual_DDMD")
    parser.add_argument("--use_mirror_loss", type=int, default=1)
    parser.add_argument("--denoiser_name", type=str, default="dncnn")
    parser.add_argument("--prior_name", type=str, default="RED")
    parser.add_argument("--n_layers", type=int, default = 10)
    parser.add_argument("--wandb_resume_id", type=str, default="")
    parser.add_argument("--lr_scheduler", type=str, default="multistep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--stepsize_init", type=float, default=0.01)
    parser.add_argument("--lamb_init", type=float, default=1.0)
    parser.add_argument("--sigma_denoiser_init", type=float, default=0.008)
    parser.add_argument("--distribute", type=int, default=0)
    parser.add_argument("--max_num_images", type=int, default=1e6)
    parser.add_argument("--noise_level_min", type=float, default=0.)
    parser.add_argument("--noise_level_max", type=float)
    parser.add_argument("--strong_convexity_backward", type=float, default=1.)
    parser.add_argument("--strong_convexity_forward", type=float, default=1.)
    parser.add_argument("--strong_convexity_potential", type=str, default='L2')
    parser.add_argument("--use_NoLip_loss", type=int, default=0)
    parser.add_argument("--eps_jacobian_loss", type=float, default=0.05)
    parser.add_argument("--jacobian_loss_weight", type=float, default=0.01)
    parser.add_argument("--max_iter_power_it", type=int, default = 10)
    parser.add_argument("--tol_power_it", type=float, default = 1e-3)
    args = parser.parse_args()

    test_only = False if args.test_only == 0 else True
    grayscale = False if args.grayscale == 0 else True
    distribute = False if args.distribute == 0 else True
    use_mirror_loss = False if args.use_mirror_loss == 0 else True
    wanddb_resume_id = None if args.wandb_resume_id == "" else args.wandb_resume_id
    lr_scheduler = None if args.lr_scheduler == "" else args.lr_scheduler
    epochs = None if args.epochs == 0 else args.epochs
    use_NoLip_loss = False if args.use_NoLip_loss == 0 else True

    if args.noise_level_max is None:
        if args.noise_model.lower() == 'gaussian':
            args.noise_level_max = 0.2
        elif args.noise_model.lower() == 'poisson':
            args.noise_level_max = 0.05
        else:
            raise ValueError('noise model not available')

    train_model(
        n_layers = args.n_layers,
        test_only = test_only,
        ckpt_pretrained = args.ckpt_pretrained,
        data_fidelity=args.data_fidelity,
        noise_model=args.noise_model,
        model_name=args.model_name,
        prior_name=args.prior_name,
        denoiser_name=args.denoiser_name,
        stepsize_init=args.stepsize_init,
        lamb_init=args.lamb_init,
        sigma_denoiser_init=args.sigma_denoiser_init,
        distribute=distribute,
        epochs=epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        patch_size=args.patch_size,
        seed=args.seed,
        gpu_num=args.gpu_num,
        lr=args.lr,
        max_num_images=args.max_num_images,
        use_mirror_loss=use_mirror_loss,
        noise_level_min=args.noise_level_min,
        noise_level_max=args.noise_level_max,
        strong_convexity_backward=args.strong_convexity_backward,
        strong_convexity_forward=args.strong_convexity_forward,
        strong_convexity_potential=args.strong_convexity_potential,
        use_NoLip_loss = args.use_NoLip_loss,
        eps_jacobian_loss = args.eps_jacobian_loss, 
        jacobian_loss_weight = args.jacobian_loss_weight, 
        max_iter_power_it = args.max_iter_power_it, 
        tol_power_it=args.tol_power_it,
        degradation=args.degradation,
        args = args
    )