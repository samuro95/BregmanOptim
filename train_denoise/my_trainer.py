import torch
import torch.utils
import torchvision
import deepinv as dinv
from deepinv.utils import plot
import wandb
from utils.utils import rescale_img

class MyTrainer(dinv.training.Trainer):
    def __init__(self, *args, **kwargs):
        super(MyTrainer, self).__init__(*args, **kwargs)

    def to_image(self, x):
        r"""
        Convert the tensor to an image. Necessary for complex images (2 channels)

        :param torch.Tensor x: input tensor
        :return: image
        """
        if x.shape[1] == 2:
            out = torch.moveaxis(x, 1, -1).contiguous()
            out = torch.view_as_complex(out).abs().unsqueeze(1)
        else:
            out = x
        return out

    def compute_metrics(
        self, x, x_net, y, physics, logs, train=True, epoch: int = None
    ):
        # Compute the metrics over the batch
        with torch.no_grad():
            for k, l in enumerate(self.metrics):
                metric = l(
                    x_net=x_net,
                    x=x,
                    y=y,
                    physics=physics,
                    model=self.model,
                    epoch=epoch,
                )

                current_log = (
                    self.logs_metrics_train[k] if train else self.logs_metrics_eval[k]
                )
                current_log.update(metric.detach().cpu().numpy())
                logs[l.__class__.__name__] = current_log.avg

                if not train and self.compare_no_learning:
                    x_lin = self.no_learning_inference(y, physics)
                    metric = l(x=x, x_net=x_lin, y=y, physics=physics, model=self.model)
                    self.logs_metrics_linear[k].update(metric.detach().cpu().numpy())
                    logs[f"{l.__class__.__name__} no learning"] = (
                        self.logs_metrics_linear[k].avg
                    )
        return logs


    def prepare_images(self, physics_cur, x, y, x_net):
        r"""
        Prepare the images for plotting.

        It prepares the images for plotting by rescaling them and concatenating them in a grid.

        :param deepinv.physics.Physics physics_cur: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Reconstruction network output.
        :returns: The images, the titles, the grid image, and the caption.
        """
        with torch.no_grad():
            if len(y.shape) == len(x.shape) and y.shape != x.shape:
                y = torch.nn.functional.interpolate(y, size=x.shape[2])
            if hasattr(physics_cur, "A_adjoint"):
                imgs = [y, physics_cur.A_adjoint(y), x_net, x]
                caption = (
                    "From top to bottom: input, backprojection, output, target"
                )
                titles = ["Input", "Backprojection", "Output", "Target"]
            else:
                imgs = [y, x_net, x]
                titles = ["Input", "Output", "Target"]
                caption = "From top to bottom: input, output, target"

            # Concatenate the images along the batch dimension
            for i in range(len(imgs)):
                imgs[i] = self.to_image(imgs[i])

            vis_array = torch.cat(imgs, dim=0)
            for i in range(len(vis_array)):
                vis_array[i] = rescale_img(vis_array[i], rescale_mode="min_max")
            grid_image = torchvision.utils.make_grid(vis_array, nrow=y.shape[0])

        return imgs, titles, grid_image, caption

    def plot(self, epoch, physics, x, y, x_net, train=True):
        r"""
        Plot the images.

        It plots the images at the end of each epoch.

        :param int epoch: Current epoch.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        """
        post_str = "Training" if train else "Eval"
        if self.plot_images and ((epoch + 1) % self.freq_plot == 0):
            imgs, titles, grid_image, caption = self.prepare_images(
                physics, x, y, x_net
            )

            # normalize the grid image
            # grid_image = rescale_img(grid_image, rescale_mode="min_max")

            # if MRI in class name, rescale = min-max
            if "MRI" in str(physics):
                rescale_mode = "min_max"
            else:
                rescale_mode = "clip"
            plot(
                imgs,
                titles=titles,
                show=self.plot_images,
                return_fig=True,
                rescale_mode=rescale_mode,
            )

            if self.wandb_vis:
                log_dict_post_epoch = {}
                images = wandb.Image(
                    grid_image,
                    caption=caption,
                )
                log_dict_post_epoch[post_str + " samples"] = images
                log_dict_post_epoch["step"] = epoch
                wandb.log(log_dict_post_epoch)
