from models.gan1d import GAN_model, Conv1dModel, LayeredGenerator, LayeredDiscriminator
import torch.nn as nn
from models.utils import get_layered_mask


def get_channels_list(args, dataset, neighbour_list):
    n_channels = dataset.n_channels

    joint_num = len(neighbour_list)

    base_channel = -1 if -1 != -1 else 128
    n_layers = -1 if -1 != -1 else 4
    if 0:
        base_channel = n_channels

    channels_list = [n_channels]
    for i in range(n_layers - 1):
        channels_list.append(base_channel * (2 ** ((i+1) // 2)))
    channels_list += [n_channels]
    # channels_list = [n_channels, base_channel, 2*base_channel, 2*base_channel, n_channels]
    if 1:
        channels_list = [((n - 1) // joint_num + 1) * joint_num for n in channels_list]
    if 0:
        factor = [1, 1, 2, 2, 1]
        channels_list = [n_channels * f for f in factor]

    return channels_list


def get_group_list(args, num_stages):
    group_list = []
    for i in range(0, num_stages, 2):
        group_list.append(list(range(i, min(i + 2, num_stages))))
    return group_list


def create_layered_model(args, dataset, evaluation=False, channels_list=None):
    if 0:
        # In new implementation the layered model has been replaced by normal model.
        return create_model(args, dataset, evaluation, channels_list)
    n_channels = len(utils.get_layered_mask('locrot', dataset.n_rot))

    neighbour_list = dataset.bvh_file.get_neighbor(threshold=2, enforce_lower=0,
                                                   enforce_contact=1)

    channels_list_layered = [n_channels, n_channels, n_channels * 2, n_channels * 2, n_channels]
    channels_list_regular = get_channels_list(args, dataset, neighbour_list) if channels_list is None else channels_list

    if len('') != 0:
        layered_gen = None
    elif 0:
        layered_gen = Conv1dModel(channels_list_regular, 5, last_active=None,
                                  padding_mode='reflect',
                                  batch_norm=0,
                                  neighbour_list=neighbour_list, skeleton_aware=0).to('cuda:0')
    else:
        layered_gen = Conv1dModel(channels_list_layered, 5, last_active=None, padding_mode='reflect',
                                  batch_norm=0,
                                  neighbour_list=None, skeleton_aware=False).to('cuda:0')

    regular_gen = Conv1dModel(channels_list_regular, 5, last_active=None, padding_mode='reflect',
                              batch_norm=0,
                              neighbour_list=neighbour_list, skeleton_aware=0).to('cuda:0')

    layered_gen = LayeredGenerator(args, layered_gen, regular_gen, dataset.n_rot, default_requires_mask=True)

    if evaluation:
        return layered_gen
    else:
        disc = Conv1dModel(channels_list_regular[:-1] + [1, ], 5, last_active=None,
                           padding_mode='reflect', batch_norm=0,
                           neighbour_list=neighbour_list, skeleton_aware=0).to('cuda:0')

        if 0:
            disc_layered = Conv1dModel(channels_list_layered[:-1] + [1, ], 5, last_active=None,
                                       padding_mode='reflect', batch_norm=0,
                                       neighbour_list=None,
                                       skeleton_aware=False).to('cuda:0')

            disc = LayeredDiscriminator(args, disc_layered, disc, dataset.n_rot)

        gan_model = GAN_model(layered_gen, disc, args, dataset)
        return layered_gen, disc, gan_model


def create_model(args, dataset, evaluation=False, channels_list=None):
    gen_last_active = None

    neighbour_list = dataset.bvh_file.get_neighbor(threshold=2, enforce_contact=1)
    if channels_list is None:
        channels_list = get_channels_list(args, dataset, neighbour_list)

    if not 0:
        print('Channel list:', channels_list)

    gen = Conv1dModel(channels_list, 5, last_active=gen_last_active,
                      padding_mode='reflect', batch_norm=0,
                      neighbour_list=neighbour_list, skeleton_aware=0).to('cuda:0')
    if evaluation:
        return gen
    else:
        disc = Conv1dModel(channels_list, 5, last_active=None,
                           padding_mode='reflect', batch_norm=0,
                           neighbour_list=neighbour_list, skeleton_aware=0).to('cuda:0')
        #gan_model = GAN_model(gen, disc, args, dataset)
        #print('???')
        return gen, disc
