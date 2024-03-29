import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, gridspec
import os
import glob
from time import strftime, localtime
from shutil import copy
# from scipy.misc import imresize
from dataset.motion import MotionData
from models.architecture import draw_example
from models.utils import get_interpolator
import torch


def read_data(conf):
    input_images = [read_shave_tensorize(path, conf.must_divide) for path in conf.input_image_path]
    return input_images


def read_shave_tensorize(path, must_divide):
    input_np = (np.array(Image.open(path).convert('RGB')) / 255.0)

    input_np_shaved = input_np[:(input_np.shape[0] // must_divide) * must_divide, :(input_np.shape[1] // must_divide) *
                               must_divide, :]

    input_tensor = im2tensor(input_np_shaved)

    return input_tensor


def tensor2im(image_tensors, imtype=np.uint8):

    if not isinstance(image_tensors, list):
        image_tensors = [image_tensors]

    image_numpys = []
    for image_tensor in image_tensors:
        # Note that tensors are shifted to be in [-1,1]
        image_numpy = image_tensor.detach().cpu().float().numpy()

        if np.ndim(image_numpy) == 4:
            image_numpy = image_numpy.transpose((0, 2, 3, 1))

        image_numpy = np.round((image_numpy.squeeze(0) + 1) / 2.0 * 255.0)
        image_numpys.append(image_numpy.astype(imtype))

    if len(image_numpys) == 1:
        image_numpys = image_numpys[0]

    return image_numpys


def im2tensor(image_numpy, int_flag=False, device=torch.device('cuda')):
    # the int flag indicates whether the input image is integer (and [0,255]) or float ([0,1])
    if int_flag:
        image_numpy /= 255.0
    # Undo the tensor shifting (see tensor2im function)
    transformed_image = np.transpose(image_numpy, (2, 0, 1)) * 2.0 - 1.0
    return torch.FloatTensor(transformed_image).unsqueeze(0).to(device)


def random_size(
    orig_size,
    curriculum=True,
    i=None,
    iter_for_max_range=None,
    must_divide=8.0,
    min_scale=0.25,
    max_scale=2.0,
    max_transform_magniutude=0.3
):
    cur_max_scale = 1.0 + (max_scale - 1.0) * np.clip(1.0 * i / iter_for_max_range, 0, 1) if curriculum else max_scale
    cur_min_scale = 1.0 + (min_scale - 1.0) * np.clip(1.0 * i / iter_for_max_range, 0, 1) if curriculum else min_scale
    cur_max_transform_magnitude = (
        max_transform_magniutude *
        np.clip(1.0 * i / iter_for_max_range, 0, 1) if curriculum else max_transform_magniutude
    )

    # set random transformation magnitude. scalar = affine, pair = homography.
    random_affine = -cur_max_transform_magnitude + 2 * cur_max_transform_magnitude * np.random.rand(2)

    # set new size for the output image
    new_size = np.array(orig_size) * (cur_min_scale + (cur_max_scale - cur_min_scale) * np.random.rand(2))
    

    return tuple(np.uint32(np.ceil(new_size * 1.0 / must_divide) * must_divide)), random_affine


def image_concat(g_preds, d_preds=None, size=None):
    hsize = g_preds[0].shape[0] + 6 if size is None else size[0]
    results = []
    if d_preds is None:
        d_preds = [None] * len(g_preds)
    for g_pred, d_pred in zip(g_preds, d_preds):
        # noinspection PyUnresolvedReferences
        dsize = g_pred.shape[1] if size is None or size[1] is None else size[1]
        result = np.ones([(1 + (d_pred is not None)) * hsize, dsize, 3]) * 255
        if d_pred is not None:
            img = (np.concatenate([d_pred] * 1, 2) - 128) * 2
            import cv2
            d_pred_new = cv2.resize(img, dsize=g_pred.shape[0:2][::-1], interpolation=cv2.INTER_NEAREST)
            con = np.concatenate([g_pred[0], d_pred_new], 0)
            result[hsize - g_pred.shape[0]:hsize + g_pred.shape[0], :g_pred.shape[1], :] = con
        else:
            result[hsize - g_pred.shape[0]:, :, :] = g_pred
        results.append(np.uint8(np.round(result)))

    return np.concatenate(results, 1)

def read_bvh(image_path):
    m = MotionData(image_path, padding=1,use_velo=1,contact=1, keep_y_pos=1, joint_reduction=1, repr='repr6d')
    x = m.raw_motion
    return x,m


def save_image(input_path,image_tensor, image_path,conf,obj):
    interpolator = get_interpolator(conf)
    #image_tensor = interpolator(image_tensor[0], size=image_tensor.shape[3]) 
    real,motion_data = read_bvh(input_path)
    #x = torch.rand_like(image_tensor) - 0.5 * 2.0 / 255.0
    #imgs = draw_example([obj.gan.G], 'random', torch.stack([x])[0][0],[x.shape[3]], x, 1, conf, all_img=True, conds=None, full_noise=0, given_noise=[x])               
    motion_data.write(image_path, image_tensor[0])


def get_scale_weights(i, max_i, start_factor, input_shape, min_size, num_scales_limit, scale_factor):
    num_scales = np.min(
        [np.int(np.ceil(np.log(np.min(input_shape) * 1.0 / min_size) / np.log(scale_factor))), num_scales_limit]
    )

    factor = start_factor**((max_i - i) * 1.0 / max_i)

    un_normed_weights = factor**np.arange(num_scales)
    weights = un_normed_weights / np.sum(un_normed_weights)
    #
    # np.clip(i, 0, max_i)
    #
    # un_normed_weights = np.exp(-((np.arange(num_scales) - (max_i - i) * num_scales * 1.0 / max_i) ** 2) / (2 * sigma ** 2))
    # weights = un_normed_weights / np.sum(un_normed_weights)

    return weights


class Visualizer:
    def __init__(self, gan, conf, test_inputs):
        self.gan = gan
        self.conf = conf
        self.G_loss = [None] * conf.max_iters
        self.D_loss_real = [None] * conf.max_iters
        self.D_loss_fake = [None] * conf.max_iters
        self.D_loss = [None] * conf.max_iters
        self.losses_GP = [None] * conf.max_iters

        self.test_inputs = test_inputs
        self.test_input_sizes = [test_input.shape[2:] for test_input in test_inputs]

        if conf.reconstruct_loss_stop_iter > 0:
            self.Rec_loss = [None] * conf.max_iters

    def recreate_fig(self):
        self.fig = plt.figure(figsize=(50, 25))
        gs = gridspec.GridSpec(9,8)
        self.gan_loss = self.fig.add_subplot(gs[0:4, 0:8])
        self.reconstruct_loss = self.fig.add_subplot(gs[5:9, 0:8])

        # First plot data
        #self.plot_gan_loss = self.gan_loss.plot([], [], 'b-', [], [], 'c--', [], [], 'r--', [], [], 'g--')
        #self.gan_loss.legend(('Generator loss', 'Discriminator loss (real)', 'Discriminator loss (fake)'))
        
        self.plot_gan_loss = self.gan_loss.plot([], [], 'b-')
        self.gan_loss.legend(('Generator loss'))
        
        #self.gan_loss.set_ylim(0,1)
        self.gan_loss.set_ylim(0,10000)

        if self.conf.reconstruct_loss_stop_iter > 0:
            self.plot_reconstruct_loss = self.reconstruct_loss.semilogy([], [])

        # Set titles
        self.gan_loss.set_title('Gan Losses')
        self.reconstruct_loss.set_title('Reconstruction Loss')
        
    def recreate_fig_GP(self):
        self.fig = plt.figure(figsize=(50, 25))
        gs = gridspec.GridSpec(9,8)
        self.plot_gp = self.fig.add_subplot(gs[0:9, 0:8])

        # First plot data
        #self.plot_gp_loss = self.plot_gp.plot([], [], 'g-', [], [], 'r--')
        #self.plot_gp.legend(('Gradient Penalty','Discriminator loss'))
        
        self.plot_gp_loss = self.plot_gp.plot([], [], 'g-')
        self.plot_gp.legend(('Discriminator loss'))
        self.plot_gp.set_ylim(0,10000)


        # Set titles
        self.plot_gp.set_title('Gradient Penalty')


    def test_and_display(self, input_path, i):
        if not i % self.conf.print_freq and i > 0:
            self.G_loss[i - self.conf.print_freq:i] = self.gan.losses_G_gan.detach().cpu().float().numpy().tolist()
            #self.D_loss_real[i -
            #                 self.conf.print_freq:i] = self.gan.losses_D_real.detach().cpu().float().numpy().tolist()
            #self.D_loss_fake[i -
            #                 self.conf.print_freq:i] = self.gan.losses_D_fake.detach().cpu().float().numpy().tolist()
            self.D_loss[i -
                             self.conf.print_freq:i] = self.gan.losses_D.detach().cpu().float().numpy().tolist()                 
            #self.losses_GP[i -
            #                 self.conf.print_freq:i] = self.gan.losses_GP.detach().cpu().float().numpy().tolist()                 
            if self.conf.reconstruct_loss_stop_iter > i:
                self.Rec_loss[i - self.conf.print_freq:i] = self.gan.losses_G_reconstruct.detach().cpu().float().numpy(
                ).tolist()

            """
            if self.conf.reconstruct_loss_stop_iter < i:
                print(
                    (
                        'iter: %d, G_loss: %f, D_loss (Real): %f, D_loss (Fake): %f, D_loss: %f, Gradient Penalty: %f, LR: %f' % (
                            i, self.G_loss[i - 1],self.D_loss_real[i - 1],self.D_loss_fake[i - 1],self.D_loss[i - 1] ,self.losses_GP[i - 1],
                            self.gan.lr_scheduler_G.get_lr()[0]
                        )
                    )
                )
            else:
                print(
                    (
                        'iter: %d, G_loss: %f, D_loss (Real): %f, D_loss (Fake): %f, D_loss: %f, Gradient Penalty: %f, Rec_loss: %f, LR: %f' % (
                            i, self.G_loss[i - 1], self.D_loss_real[i - 1],self.D_loss_fake[i - 1] ,self.D_loss[i - 1] ,self.losses_GP[i - 1],
                            self.Rec_loss[i - 1], self.gan.lr_scheduler_G.get_lr()[0]
                        )
                    )
                )
            """    
            if self.conf.reconstruct_loss_stop_iter < i:
                print(
                    (
                        'iter: %d, G_loss: %f, D_loss: %f, LR: %f' % (
                            i, self.G_loss[i - 1] ,self.D_loss[i - 1] ,
                            self.gan.lr_scheduler_D.get_lr()[0]
                        )
                    )
                )
            else:
                print(
                    (
                        'iter: %d, G_loss: %f, D_loss: %f, Rec_loss: %f, LR: %f' % (
                            i, self.G_loss[i - 1] ,self.D_loss[i - 1], 
                            self.Rec_loss[i - 1], self.gan.lr_scheduler_D.get_lr()[0]
                        )
                    )
                ) 

        if not i % self.conf.display_freq and i > 0:
            plt.gcf().clear()
            plt.close()
            self.recreate_fig()

            g_preds = [self.gan.input_tensor_noised, self.gan.G_pred]
            d_preds = [
                self.gan.D.forward(self.gan.input_tensor_noised.detach(), self.gan.scale_weights), self.gan.d_pred_fake
            ]
            reconstructs = self.gan.reconstruct
            input_size = self.gan.input_tensor_noised.shape[2:]

            self.plot_gan_loss[0].set_data(list(range(i)), self.G_loss[:i])
            #self.plot_gan_loss[1].set_data(list(range(i)), self.D_loss_real[:i])
            #self.plot_gan_loss[2].set_data(list(range(i)), self.D_loss_fake[:i])
            self.gan_loss.set_xlim(0, i)

            if self.conf.reconstruct_loss_stop_iter > i:
                self.plot_reconstruct_loss[0].set_data(list(range(i)), self.Rec_loss[:i])
                self.reconstruct_loss.set_ylim(np.min(self.Rec_loss[:i]), np.max(self.Rec_loss[:i]))
                self.reconstruct_loss.set_xlim(0, i)

            """
            self.d_map_real.imshow(
                self.gan.d_pred_real[0:1, :, :, :].detach().cpu().float().numpy().squeeze(),
                cmap='gray',
                vmin=0,
                vmax=1
            )
            
            if self.conf.reconstruct_loss_stop_iter > i:
                self.reconstruction.imshow(np.clip(image_concat([tensor2im(reconstructs)]), 0, 255), vmin=0, vmax=255)
            """
            plt.savefig(self.conf.output_dir_path + '/monitor_%d' % i)
            
            plt.gcf().clear()
            plt.close()
            self.recreate_fig_GP()
            
            #self.plot_gp_loss[0].set_data(list(range(i)), self.losses_GP[:i])
            self.plot_gp_loss[0].set_data(list(range(i)), self.D_loss[:i])
            self.plot_gp.set_xlim(0, i)
            plt.savefig(self.conf.output_dir_path + '/monitor_GP_%d' % i)
            
            save_image(input_path,self.gan.G_pred, self.conf.output_dir_path + '/result_iter_%d.bvh' % i,self.conf,self)


def prepare_result_dir(conf):
    # Create results directory
    conf.output_dir_path += '/' + conf.name + strftime('_%b_%d_%H_%M_%S', localtime())
    os.makedirs(conf.output_dir_path)

    # Put a copy of all *.py files in results path, to be able to reproduce experimental results
    if conf.create_code_copy:
        local_dir = os.path.dirname(__file__)
        for py_file in glob.glob(local_dir + '/*.py'):
            copy(py_file, conf.output_dir_path)
        if conf.resume:
            copy(conf.resume, os.path.join(conf.output_dir_path, 'starting_checkpoint.pth.tar'))
    return conf.output_dir_path


def homography_based_on_top_corners_x_shift(rand_h):
    p = np.array(
        [
            [1., 1., -1, 0, 0, 0, -(-1. + rand_h[0]), -(-1. + rand_h[0]), -1. + rand_h[0]],
            [0, 0, 0, 1., 1., -1., 1., 1., -1.], [-1., -1., -1, 0, 0, 0, 1 + rand_h[1], 1 + rand_h[1], 1 + rand_h[1]],
            [0, 0, 0, -1, -1, -1, 1, 1, 1], [1, 0, -1, 0, 0, 0, 1, 0, -1], [0, 0, 0, 1, 0, -1, 0, 0, 0],
            [-1, 0, -1, 0, 0, 0, 1, 0, 1], [0, 0, 0, -1, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ],
        dtype=np.float32
    )
    b = np.zeros((9, 1), dtype=np.float32)
    b[8, 0] = 1.
    h = np.dot(np.linalg.inv(p), b)
    return torch.from_numpy(h).view(3, 3).cuda()


def homography_grid(theta, size):
    r"""Generates a 2d flow field, given a batch of homography matrices :attr:`theta`
    Generally used in conjunction with :func:`grid_sample` to
    implement Spatial Transformer Networks.

    Args:
        theta (Tensor): input batch of homography matrices (:math:`N \times 3 \times 3`)
        size (torch.Size): the target output image size (:math:`N \times C \times H \times W`)
                           Example: torch.Size((32, 3, 24, 24))

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)
    """
    a = 1
    b = 1
    y, x = torch.meshgrid((torch.linspace(-b, b, np.int(size[-2] * a)), torch.linspace(-b, b, np.int(size[-1] * a))))
    n = np.int(size[-2] * a) * np.int(size[-1] * a)
    hxy = torch.ones(n, 3, dtype=torch.float)
    hxy[:, 0] = x.contiguous().view(-1)
    hxy[:, 1] = y.contiguous().view(-1)
    out = hxy[None, ...].cuda().matmul(theta.transpose(1, 2))
    # normalize
    out = out[:, :, :2] / out[:, :, 2:]
    return out.view(theta.shape[0], np.int(size[-2] * a), np.int(size[-1] * a), 2)


def hist_match(source, template, mask_3ch):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source_masked = source.ravel()[mask_3ch.ravel() > 128]
    template = template.ravel()
    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source_masked, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    out = source.copy().ravel()
    out[mask_3ch.ravel() > 128] = interp_t_values[bin_idx]
    return out.reshape(oldshape)
