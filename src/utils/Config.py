#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：Config.py
@Author     ：Heywecome
@Date       ：2024/3/28 14:45 
@Description：The main class of Config
"""
import re
import os
import yaml
import paddle
import logging


class Config(object):
    def __init__(self, model=None, dataset=None, config_dict=None):
        """
        Args:
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        if config_dict is None:
            config_dict = {}
        config_dict['model'] = model
        config_dict['dataset'] = dataset
        # model type
        self.final_config_dict = self._load_dataset_model_config(config_dict, mg)
        # cmd and main.py will choose the final config
        self.final_config_dict.update(config_dict)
        self._set_default_parameters()
        self._init_device()

    def _load_dataset_model_config(self, config_dict):
        """
        load config from yaml files
        """
        file_config_dict = dict()
        file_list = []
        cur_dir = os.path.join(os.getcwd(), 'configs')
        file_list.append(os.path.join(cur_dir, 'overall.yaml'))
        file_list.append(os.path.join(cur_dir, 'dataset', "{}.yaml".format(config_dict['dataset'])))
        file_list.append(os.path.join(cur_dir, 'model', '{}.yaml'.format(config_dict['model'])))

        hyper_parameters = []
        for file in file_list:
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    fdata = yaml.load(f.read(), loader=self._build_yaml_loader())
                    if fdata.get('hyper_parameters'):
                        hyper_parameters.extend(fdata['hyper_parameters'])
                    file_config_dict.update(fdata)

        file_config_dict['hyper_parameters'] = hyper_parameters
        return file_config_dict

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader

    def _set_default_parameters(self):
        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True
        # if seed not in hyper_parameters, then add
        if "seed" not in self.final_config_dict['hyper_parameters']:
            self.final_config_dict['hyper_parameters'] += ['seed']

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = paddle.device.set_device("cuda"
                                                                    if paddle.is_compiled_with_cuda() and use_gpu
                                                                    else "cpu")
        logging.info("Device:", self.final_config_dict['device'])

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = '\n'
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()
