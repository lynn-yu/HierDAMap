import os
import sys
from data.nus_mapping import MappingNuscData,FlipMappingNuscData
from data.data_av2 import AV2Dataset,AV2Dataset_source
import torch
import numpy as np
from nuscenes.nuscenes import NuScenes

def worker_rnd_init(x):
    np.random.seed(13 + x)
def compile_data_domain_av2(version, dataroot, data_aug_conf, grid_conf, nsweeps, domain_gap, source, target, bsz, nworkers):



    if version=='v1.0-mini':
        ttraindata = AV2Dataset(root_path='/data1/lsy/argoverse', split='test', data_conf=data_aug_conf, grid_conf=grid_conf, is_train=True)
        valdata = AV2Dataset(root_path='/data1/lsy/argoverse', split='val', data_conf=data_aug_conf, grid_conf=grid_conf, is_train=False)
    else:
        ttraindata = AV2Dataset(root_path='/data1/lsy/argoverse', split='train', data_conf=data_aug_conf, grid_conf=grid_conf, is_train=True)
        valdata =   AV2Dataset(root_path='/data1/lsy/argoverse', split='test', data_conf=data_aug_conf,grid_conf=grid_conf,is_train=False)



    ttrainloade = torch.utils.data.DataLoader(ttraindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return  ttrainloade, valloader

def compile_data_domain_n2a(version, dataroot, data_aug_conf, grid_conf, nsweeps, domain_gap, source, target, bsz, nworkers):
    nusc = NuScenes(version=version,
                    dataroot=dataroot,
                    verbose=False)

    straindata = MappingNuscData(nusc, is_train=True, data_aug_conf=data_aug_conf,
                                 grid_conf=grid_conf, nsweeps=nsweeps,
                                 domain_gap=domain_gap, domain=source, domain_type='strain')


    if version=='v1.0-mini':
        ttraindata = AV2Dataset(root_path='/data1/lsy/argoverse', split='test', data_conf=data_aug_conf, grid_conf=grid_conf, is_train=True)
        valdata = AV2Dataset_source(root_path='/data1/lsy/argoverse', split='val', data_conf=data_aug_conf, grid_conf=grid_conf, is_train=False)
    else:
        ttraindata = AV2Dataset(root_path='/data1/lsy/argoverse', split='train', data_conf=data_aug_conf, grid_conf=grid_conf, is_train=True)
        valdata =   AV2Dataset_source(root_path='/data1/lsy/argoverse', split='test', data_conf=data_aug_conf,grid_conf=grid_conf,is_train=False)


    strainloader = torch.utils.data.DataLoader(straindata, batch_size=bsz,
                                               shuffle=True,
                                               num_workers=nworkers,
                                               drop_last=True,
                                               worker_init_fn=worker_rnd_init)
    ttrainloade = torch.utils.data.DataLoader(ttraindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return strainloader, ttrainloade, valloader

def compile_data_domain_a2n(version, dataroot, data_aug_conf, grid_conf, nsweeps, domain_gap, source, target, bsz, nworkers):
    nusc = NuScenes(version=version,
                    dataroot=dataroot,
                    verbose=False)
    straindata = AV2Dataset_source(root_path='/data1/lsy/argoverse', split='train', data_conf=data_aug_conf, grid_conf=grid_conf, is_train=True)
    ttraindata = FlipMappingNuscData(nusc, is_train=True, data_aug_conf=data_aug_conf,
                                 grid_conf=grid_conf, nsweeps=nsweeps,
                                 domain_gap=domain_gap, domain=target, domain_type='ttrain')
    valdata = MappingNuscData(nusc, is_train=False, data_aug_conf=data_aug_conf,
                                 grid_conf=grid_conf, nsweeps=nsweeps,
                                 domain_gap=domain_gap, domain=target, domain_type='tval')



    strainloader = torch.utils.data.DataLoader(straindata, batch_size=bsz,
                                               shuffle=True,
                                               num_workers=nworkers,
                                               drop_last=True,
                                               worker_init_fn=worker_rnd_init)
    ttrainloade = torch.utils.data.DataLoader(ttraindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return strainloader, ttrainloade, valloader