from configs import Config
from InGAN import InGAN
import os
from util import Visualizer, read_data
from traceback import print_exc
import numpy as np
import torch
from dataset.motion import MotionData, load_multiple_dataset
from models import create_model, create_layered_model, get_group_list
from models.architecture import get_pyramid_lengths, joint_train
from models.utils import get_interpolator
from option import TrainOptionParser
from os.path import join as pjoin
import time
from torch.utils.tensorboard import SummaryWriter



# Load configuration
conf = Config().parse()
# Prepare data
input_images1 = MotionData(conf.input_image_path[0],padding=1,use_velo=1,contact=1, keep_y_pos=1, joint_reduction=1, repr='repr6d')
gen, input_images = create_model(conf, input_images1, evaluation=False)
input_images = [input_images(input_images1.raw_motion.cuda())]
input_images = torch.stack(input_images)                            

# Create complete model
gan = InGAN(conf)
# If required, fine-tune from some checkpoint
if conf.resume is not None:
    gan.resume(os.path.join(conf.resume))

# Define visualizer to monitor learning process
visualizer = Visualizer(gan, conf, [input_images])

# Main training loop
for i in range(conf.max_iters + 1):

    # Train a single iteration on the current data instance
    try:
        gan.train_one_iter(i, input_images)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print('Something went wrong in iteration %d, While training.' % i)
        print_exc()

    # Take care of all testing, saving and presenting of current results and status
    try:
        visualizer.test_and_display(conf.input_image_path[0],i)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print('Something went wrong in iteration %d, While testing or visualizing.' % i)
        print_exc()

    # Save snapshot when needed
    try:
        if i > 0 and not i % conf.save_snapshot_freq:
            gan.save(os.path.join(conf.output_dir_path, 'checkpoint_%07d.pth.tar' % i))
            del gan
            gan = InGAN(conf)
            gan.resume(os.path.join(conf.output_dir_path, 'checkpoint_%07d.pth.tar' % i))
            visualizer.gan = gan
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print('Something went wrong in iteration %d, While saving snapshot.' % i)
        print_exc()
