#!/usr/bin/env python
from __future__ import print_function
import streamlit as st
import time
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
# #Design
# st.set_page_config(page_title='HVT_HVKTQS',page_icon="☀", layout="wide")
# st.header('REVERSE ENGINEERING NEURAL NETWORK')
# st.snow()
# # df = pd.DataFrame(
# #    np.random.randn(10, 5),
# #    columns=('col %d' % i for i in range(5)))

# # st.table(df)

# # with st.spinner('Wait for it...'):
# #     time.sleep(1)
# chart_data = pd.DataFrame(
#     np.random.randn(50, 3),
#     columns=["a", "b", "c"])

# st.bar_chart(chart_data)



__author__ = 'joon'


import sys

sys.path.insert(0, 'src')

from lib_pytorch.baseblocks import *
from imports.basic_modules import *
from imports.ResearchTools import *
from imports.pytorch_imports import *
####
from mnist_nets.train_mnist_nets import main as build_mnist_nets
from mnist_nets.postprocess_mnist_nets import main as postprocess_mnist_nets
from mnist_metamodel.mnist_metamodel import config as config_metamodel
###

from mnist_nets.mnist import mnist_data_transform, load_model
from mnist_nets.mnist import config as mnist_config
from lib_pytorch.basetool import ListDataset, TrainCurve


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def label_generator(attributes):
    labels = {}
    for ky in attributes.keys():
        if isinstance(attributes[ky], dict):
            labels[ky] = label_generator(attributes[ky])
        else:
            attr = attributes[ky]
            attr = sorted(attr)
            labels[ky] = dict(zip(attr, range(len(attributes[ky]))))
    return labels


def config(control, conf):
    co = experiment_control(
        control,
        conf,
        dict(
            data=dict(
                name='dnet10000',
                subset=0,
                eval=1000,
            ),
            seed=0,
            i=dict(
                init='randval',
                noise='U1',
                opt=dict(
                    optimiser='SGD',
                    weight_decay=0.0,
                    batch_size=10,
                ),
            ),
            m=dict(
                name='linearnet',
                opt=dict(
                    optimiser='SGD',
                    weight_decay=0.01,
                    batch_size=0,
                ),
            ),
            opt=dict(
                epochs=1000,
            ),
            setup=dict(
                qseed=0,
                outrep='score',
            ),
        ),
        exclude=[
            dict(condition=dict(method='m'), remove='i'),
            dict(condition=dict(method='m'), remove='opt/sequence'),
            dict(condition=dict(method='i'), remove='m'),
            dict(condition=dict(method='i'), remove='opt/sequence'),
            dict(condition=dict(method='i'), remove='setup/outrep'),
            dict(condition=dict(method='i'), remove='setup/nquery'),
            dict(condition=dict(method='i'), remove='setup/qseed'),
        ],
    )

    if co.control['method'] == 'i':
        assert (co.control['setup']['outrep'] == 'argmax')
        assert (co.control['setup']['nquery'] == 1)
        assert (co.control['setup']['qseed'] == 0)

    set_seed(co.control['seed'])
    if conf['gpu'] is not None:
        torch.cuda.set_device(conf['gpu'])

    co.conf['attributes'] = dict(
        net=dict(
            act=['relu', 'elu', 'prelu', 'tanh'],
            drop=['normal', 'none'],
            pool=['max_2', 'none'],
            ks=[3, 5],
            n_conv=[2, 3, 4],
            n_fc=[2, 3, 4],
        ),
        opt=dict(
            optimiser=['SGD', 'ADAM', 'RMSprop'],
            batch_size=[256, 128, 64],
        ),
        data=dict(
            subset=['all', 'half_0', 'half_1', 'quarter_0', 'quarter_1', 'quarter_2', 'quarter_3'],
        ),
        etc=dict(
            data_size=['all', 'half', 'quarter'],
            ens=[0, 1],
            n_param=range(14, 22, 1),
        )
    )

    co.conf['targetlab'] = label_generator(co.conf['attributes'])
    co.filemanager('pima.pickle.dat', 'finalmodel')
    precomp_ignores = ['i', 'm', 'opt', 'setup/target', 'setup/outrep']
    co.filemanager('precomp_train.pth.tar', 'precomp_train', ignore=precomp_ignores)
    co.filemanager('precomp_test.pth.tar', 'precomp_test', ignore=precomp_ignores)

    return co


class mlp(nn.Module):
    def __init__(self, netname, nquery, queryoutdim, target, gpu=True):
        super(mlp, self).__init__()
        _, self.nlayer, self.hiddendim = netname.split('_')
        self.nlayer = int(self.nlayer)
        self.hiddendim = int(self.hiddendim)
        self.target = target
        self.gpu = gpu

        self.inputdim = queryoutdim * nquery
        self.act = F.relu
        self.fc1 = nn.Linear(self.inputdim, self.hiddendim)

        fc_iter = []
        for n in range(self.nlayer - 2):
            fc_iter.append(linearblock(self.hiddendim, self.hiddendim))
        self.fc_iter = nn.Sequential(*fc_iter)
        self.fc_final = self._layer_generator(target)

    def forward(self, x, return_type='structured_output'):
        x = x.view(-1, self.inputdim)
        x = self.act(self.fc1(x))
        x = self.fc_iter(x)
        if return_type == 'structured_output':
            return self._apply_fc_layers(self.fc_final, self.target, x)
        elif return_type == 'cat_output':
            x = self._apply_fc_layers(self.fc_final, self.target, x)
            out = []
            self._to_list(x, out)
            return torch.cat(out, 1)
        elif return_type == 'embedding':
            return x
        else:
            raise ValueError('Return_type must be one of {structured_output, cat_output, embedding}.')

    def _layer_generator(self, attributes):
        layers = {}
        for ky in attributes.keys():
            if isinstance(attributes[ky], dict):
                layers[ky] = self._layer_generator(attributes[ky])
            else:
                attr = attributes[ky]
                layers[ky] = nn.Linear(self.hiddendim, len(attr))
                if self.gpu is not None:
                    layers[ky].cuda()
        return layers

    def _apply_fc_layers(self, fc_final, target, x):
        x_out = {}
        for ky in fc_final.keys():
            if isinstance(fc_final[ky], dict):
                x_out[ky] = self._apply_fc_layers(fc_final[ky], target[ky], x)
            else:
                x_out[ky] = F.log_softmax(fc_final[ky](x), dim=1)
        return x_out

    def _to_list(self, structured_x, out):
        for ky in structured_x.keys():
            if isinstance(structured_x[ky], dict):
                self._to_list(structured_x[ky], out)
            else:
                out.append(structured_x[ky])


def initstats_generator(attributes, mat=False):
    labels = {}
    for ky in attributes.keys():
        if isinstance(attributes[ky], dict):
            labels[ky] = initstats_generator(attributes[ky], mat=mat)
        else:
            if mat:
                labels[ky] = np.zeros((len(attributes[ky]), len(attributes[ky])), dtype=np.int)
            else:
                labels[ky] = 0
    return copy.deepcopy(labels)


class inputPerturber(object):
    def __init__(self, co):
        self.pert, self.qlabel = self.__init_pert(co.control['i']['init'], co.control['setup']['nquery'],
                                                  co.control['setup']['qseed'])
        self.__set_clip(co.control['i']['clip'])
        self.__set_noise(co.control['i']['noise'])
        self.__lr = co.control['i']['opt']['lr']

    def __init_pert(self, inittype, npert, qseed=0):
        if inittype == 'black':
            pert = 0. * np.ones((npert, 1, 28, 28), dtype=np.float32)
            pert = mnist_data_transform(pert)
        elif inittype == 'gray':
            pert = .5 * np.ones((npert, 1, 28, 28), dtype=np.float32)
            pert = mnist_data_transform(pert)
        elif inittype == 'white':
            pert = 1. * np.ones((npert, 1, 28, 28), dtype=np.float32)
            pert = mnist_data_transform(pert)
        elif inittype == 'unifnoise':
            pert = np.random.uniform(0.0, 1.0, (npert, 1, 28, 28)).astype(np.float32)
            pert = mnist_data_transform(pert)
        elif inittype == 'randval':
            valdata = load_from_cache('cache/mnist_val.pkl', python23_conversion=True)['ip']
            tmp = load_from_cache('cache/mnist_val_with_label.pkl', python23_conversion=True)
            qlabel = tmp['label']
            if qseed == 0:
                sampler = load_from_cache('cache/mnist_val_queries1000.pkl', python23_conversion=True)[:npert]
            else:
                sampler = load_from_cache('cache/mnist_val_queries10000_qseed.pkl',
                                          python23_conversion=True)[qseed][:npert]
            pert = valdata[sampler].astype(np.float32)
            # im = Image.open(pert[1])

            # # ab = load_image_PIL(pert)
            # with st.expander("Ảnh Đầu Vào"):
            #     for iii in range(npert):
            #         fig = plt.figure()
            #         plt.imshow(pert[iii].reshape((28,28)), interpolation = "none",  cmap="gray")
            #         st.pyplot(fig)
            # plt.show()
            # ab = np.resize(pert[0],(28,28))
            # im = Image.fromarray(ab, 'L')
            #st.image(pert)
            qlabel = qlabel[sampler]
        else:
            raise ValueError('Inittype must be one of {gray, white, unifnoise, randval}.')

        return pert, qlabel

    def __set_clip(self, clipran):
        if isinstance(clipran, list):
            self.__clipflag = True
            self.__cliprange = [mnist_data_transform(val) for val in clipran]
        elif isinstance(clipran, float):
            self.__clipflag = True
            down, up = mnist_data_transform(0), mnist_data_transform(1)
            self.__cliprange = [np.maximum(self.pert - clipran, down), np.minimum(self.pert + clipran, up)]
        else:
            self.__clipflag = False
            self.__cliprange = None

    def __set_noise(self, noisetok):
        if noisetok == 'none':
            self.__noiseflag = False
            self.__noisefunction = None
        elif 'U' in noisetok:
            self.__noiseflag = True
            self.__noisefunction = lambda shape: np.random.uniform(-float(noisetok[1:]) / 2 / 256,
                                                                   float(noisetok[1:]) / 2 / 256,
                                                                   size=shape)

    def _add_to_pert(self, acc_dx):
        self.pert += acc_dx * self.__lr
        if self.__noiseflag:
            self.pert += self.__noisefunction(self.pert.shape)
        if self.__clipflag:
            self.pert = np.maximum(self.pert, self.__cliprange[0])
            self.pert = np.minimum(self.pert, self.__cliprange[1])


class blackBoxRevealer(inputPerturber):
    def __init__(self, co):
        super(blackBoxRevealer, self).__init__(co)
        self.co = co
        self.method = self.co.control['method']
        self._set_target()

        if 'i' in self.method:
            self.train_loader_input, self.test_loader = self.prepare_data('i')
        if 'm' in self.method:
            self._load_metanet_model()
            self._set_meta_optimiser()
            self.train_loader_metanet, self.test_loader = self.prepare_data('m')

        self._set_epochmanager()
        self.curve = TrainCurve('train_epoch', 'train_batch_index', 'train_batch_loss', 'train_batch_avgacc',
                                'test_epoch', 'test_loss', 'test_avgacc')
        self.epoch = 0

        if 'm' in self.method:
            self.query_output_fixed = {}
            self._prepare_queryoutput(phase='train')
            self._prepare_queryoutput(phase='test')

    def _get_attr_recur(self, finding, obj):
        if len(finding) <= 1:
            return obj[finding[0]]
        else:
            return self._get_attr_recur(finding[1:], obj[finding[0]])

    def _select_attribute(self, attr_dict, target):
        for ky in attr_dict.keys():
            if ky == target[0]:
                if isinstance(attr_dict[ky], dict):
                    self._select_attribute(attr_dict[ky], target[1:])
                else:
                    pass
            else:
                attr_dict.pop(ky)

    def _remove_attribute(self, attr_dict, target):
        for ky in attr_dict.keys():
            if ky == target[0]:
                if isinstance(attr_dict[ky], dict):
                    self._remove_attribute(attr_dict[ky], target[1:])
                else:
                    attr_dict.pop(ky)

    def _set_target(self):
        if self.co.control['setup']['target'] == 'all':
            self.target = copy.deepcopy(self.co.conf['attributes'])
            self.labelmapping = copy.deepcopy(self.co.conf['targetlab'])
            # if 'ex' in self.co.control['setup']['split']:
            if False:
                attrs = self.co.control['setup']['split'].split("^")[1:]
                for attr in attrs:
                    self._remove_attribute(self.target, attr.split('/'))
                    if attr.split('/')[1] in ['n_conv', 'n_fc', 'n_param']:
                        pass
                    if attr.split('/')[1] in ['data_size', 'subset']:
                        self._remove_attribute(self.target, ['etc', 'data_size'])
                self._remove_attribute(self.target, ['etc', 'ens'])
                self._remove_attribute(self.target, ['etc', 'n_param'])

        else:
            self.target = copy.deepcopy(self.co.conf['attributes'])
            self._select_attribute(self.target, self.co.control['setup']['target'].split('/'))
            self.labelmapping = self.co.conf['targetlab']

    def _set_meta_optimiser(self):
        optoptions = self.co.control['m']['opt']
        if optoptions['optimiser'] == 'SGD':
            self.metaoptimizer = optim.SGD(self.metanet.parameters(), lr=optoptions['lr'],
                                           weight_decay=optoptions['weight_decay'])
        elif optoptions['optimiser'] == 'ADAM':
            self.metaoptimizer = optim.Adam(self.metanet.parameters(), lr=optoptions['lr'],
                                            weight_decay=optoptions['weight_decay'])
        else:
            raise ValueError('Meta-optimiser should be one of {SGD, ADAM}.')

    def _load_metanet_model(self):
        nquery = self.co.control['setup']['nquery']
        metanetname = self.co.control['m']['name']
        queryoutdim = 10

        if 'mlp' in metanetname:
            # self.metanet = mlp(
            #     netname=metanetname,
            #     nquery=nquery,
            #     queryoutdim=queryoutdim,
            #     target=self.target,
            #     gpu=self.co.conf['gpu']
            # )
            self.metanet = torch.load("cache/model.pth")
        else:
            raise ValueError('Metanet only supports mlp.')

        if self.co.conf['gpu'] is not None:
            self.metanet.cuda()

    def prepare_data(self, whichnet):
        split = self.co.control['setup']['split']
        splitidtr = self.co.control['setup']['splitidtr']
        splitidte = self.co.control['setup']['splitidte']
        traindata = self.co.control['data']['name']
        trainsubset = self.co.control['data']['subset']
        testsubset = self.co.control['data']['eval']

        if traindata == 'dnet10000':
            data_cos = load_from_cache('cache/modelzoo-mnist/dnet10000_cos_pruned_ensembled.pkl',
                                       python23_conversion=True)
        else:
            raise ValueError('Meta-training only supports dnet10000.')

        traincos = []
        testcos = []

        if split == 'rand':
            splitvec = load_from_cache('cache/randsplit_%s' % traindata)
            vecs = [range(len(splitvec) // 2), range(len(splitvec) // 2, len(splitvec))]
            splitvec_train = np.array(splitvec)[np.array(vecs)[splitidtr]].reshape(-1)
            splitvec_test = np.array(splitvec)[np.array(vecs)[splitidte]].reshape(-1)

            traincos = np.array(data_cos)[splitvec_train]
            testcos = np.array(data_cos)[splitvec_test]

        elif split.split("^")[0] == 'ex':
            attrs = split.split("^")[1:]
            splitcandidates = []
            for attr in attrs:
                splitcandidates.append(
                    self._get_attr_recur(attr.split('/'), self.co.conf['attributes'])
                )
            for co in data_cos:
                trainflag = True
                testflag = True
                for attr, spcd in zip(attrs, splitcandidates):
                    this_attr = self._get_attr_recur(attr.split('/'), co.control)
                    if this_attr not in np.array(spcd)[splitidtr]:
                        trainflag = False
                    if this_attr not in np.array(spcd)[splitidte]:
                        testflag = False
                if trainflag:
                    traincos.append(co)
                if testflag:
                    testcos.append(co)
        else:
            raise ValueError('Experimental split should be one of {rand, ex^##}.')

        if trainsubset != 0:
            traincos = traincos[:trainsubset]

        if testsubset != 0:
            testcos = testcos[:testsubset]

        traindataset = ListDataset(traincos)
        testdataset = ListDataset(testcos)

        train_loader = torch.utils.data.DataLoader(traindataset,
                                                   batch_size=self.co.control[whichnet]['opt']['batch_size'],
                                                   shuffle=True, pin_memory=False, collate_fn=lambda x: x)
        test_loader = torch.utils.data.DataLoader(testdataset, batch_size=self.co.conf['test_batch_size'],
                                                  shuffle=False, pin_memory=False, collate_fn=lambda x: x)

        return train_loader, test_loader

    def _set_epochmanager(self):
        epochs = self.co.control['opt']['epochs']

        if len(self.method) == 1:
            self.epochmanager = self.method * epochs
            self.epochmanager_init_m = ['init'] * (epochs + 1)
        else:
            sequence = self.co.control['opt']['sequence']
            initseq = sequence[:-2]
            recurseq = sequence[-2:]
            if sequence[0] == 'm':
                which = ['i', 'm']
            else:
                which = ['m', 'i']

            self.epochmanager = []
            self.epochmanager_init_m = []
            n = 0
            while True:
                n += 1
                if n < len(initseq):
                    self.epochmanager += which[n % 2] * initseq[n]
                    self.epochmanager_init_m += ['init'] * initseq[n]
                else:
                    remainder = (n - len(initseq)) % 2
                    self.epochmanager += which[n % 2] * recurseq[remainder]
                    self.epochmanager_init_m += ['noin'] * recurseq[remainder]

                if len(self.epochmanager) >= epochs:
                    self.epochmanager_init_m += ['noin']
                    break

    def _compute_queryoutput(self, control, maindir, x):
        model = load_model(control, gpu=self.co.conf['gpu'])
        params = torch.load(osp.join(maindir, 'final.pth.tar'), map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])
        model.eval()
        query_output = model(x)
        # y_predict = model.predict(x.reshape(1,28,28,1))
        # print('Giá trị dự đoán: ', np.argmax(y_predict))
        return query_output

    def _compute_queryoutput_ensemble(self, co, x):
        if isinstance(co.control['seed'], int):
            query_output = self._compute_queryoutput(co.control, co.maindir, x)
        else:
            seeds = co.control['seed']

            def ct_maindir(sdidx):
                current_seed = seeds[sdidx]
                ct = copy.deepcopy(co.control)
                ct['seed'] = current_seed
                cf = copy.deepcopy(co.conf)
                co_new = mnist_config(ct, cf, called=True)
                return ct, co_new.maindir

            query_output_all = []

            for sdidx in range(len(co.control['seed'])):
                ct, maindir = ct_maindir(sdidx)
                query_output_all.append(self._compute_queryoutput(ct, maindir, x).view(1, -1, 10))

            query_output = torch.sum(torch.cat(query_output_all, dim=0), dim=0) / len(co.control['seed'])

        return query_output

    def _prepare_queryoutput(self, phase='train', suffix=''):
        try:
            self.query_output_fixed[phase] = load_from_cache(self.co.dirs['precomp_' + phase + suffix],
                                                             python23_conversion=True)
        except IOError:
            print("Precomputing query output..")
            if phase == 'train':
                tl = self.train_loader_metanet
            else:
                tl = self.test_loader

            self.query_output_fixed[phase] = {}

            for bi, cos in enumerate(tl):  # for each batch
                print('Batch: [{}/{} ({:.0f}%)]'.format(
                    bi * tl.batch_size, len(tl.dataset), 100. * bi / len(tl)
                ))
                for ii in range(tl.batch_size):
                    co = cos[ii]
                    if self.co.conf['gpu'] is not None:
                        pert = torch.from_numpy(self.pert).cuda()
                    else:
                        pert = torch.from_numpy(self.pert)

                    x = Variable(pert, requires_grad=True)
                    query_output = self._compute_queryoutput_ensemble(co, x)
                    co.update_token()
                    self.query_output_fixed[phase][co.token] = query_output.data.cpu().numpy()
            save_to_cache(self.query_output_fixed[phase], self.co.dirs['precomp_' + phase + suffix])

    def _one_hot(self, arr):
        epsilon = 1e-24
        new_arr = np.ones_like(arr) * epsilon / (arr.shape[1] - 1)
        argmax = arr.argmax(1)
        for amidx in range(len(argmax)):
            new_arr[amidx, argmax[amidx]] = 1.0 - epsilon
        return np.log(new_arr)

    def _ranking(self, arr, topk=9):
        topk = int(topk)
        factor = 10.
        new_arr = np.zeros_like(arr)
        argsort = np.argsort(arr, axis=1)[:, ::-1]
        for iii in range(len(argsort)):
            for jjj in range(1, 1 + topk):
                new_arr[iii, argsort[iii, jjj]] = jjj * 9. / topk
            for jjj in range(1 + topk, 10):
                new_arr[iii, argsort[iii, jjj]] = 9.
        return -factor * new_arr

    def _onlytop(self, arr, topk=9):
        topk = int(topk)
        factor = 10.
        new_arr = np.ones_like(arr) * 9.
        argsort = np.argsort(arr, axis=1)[:, ::-1]
        for iii in range(len(argsort)):
            new_arr[iii, argsort[iii, topk - 1]] = 0.
        return -factor * new_arr

    def _get_target(self, co, tar, return_attr=False):
        if tar[0] == 'etc':
            target_str = tar[1]
            if target_str == 'ens':
                if isinstance(co.control['seed'], int):
                    tar_this = attr = 0
                else:
                    tar_this = attr = 1
            elif target_str == 'data_size':
                attr = co.control['data']['subset'].split('_')[0]
                tar_this = self._get_attr_recur(tar, self.labelmapping)[attr]
            elif target_str == 'n_param':
                attr = int(np.floor(np.log2(co.n_param)))
                tar_this = self._get_attr_recur(tar, self.labelmapping)[attr]
        else:
            attr = self._get_attr_recur(tar, co.control)
            tar_this = self._get_attr_recur(tar, self.labelmapping)[attr]

        if return_attr:
            return attr
        else:
            return tar_this

    def _compute_loss_pred_recur(self, metaoutput, co, bs, attributes,
                                 target_tuple=[], return_loss=False, loss=[]):
        target_cpu = {}
        pred = {}
        for ky in attributes.keys():
            if isinstance(attributes[ky], dict):
                out_ = self._compute_loss_pred_recur(metaoutput[ky], co, bs, attributes[ky],
                                                     target_tuple=target_tuple + [ky], return_loss=return_loss,
                                                     loss=loss)
            else:
                out_ = self._compute_loss_pred(metaoutput[ky], co, bs, target_tuple + [ky], return_loss=return_loss)
                if return_loss:
                    loss.append(out_[2])

            target_cpu[ky] = out_[0]
            pred[ky] = out_[1]

        return target_cpu, pred

    def _compute_loss_pred(self, metaoutput, co, bs, target, return_loss=False):
        tar_this = self._get_target(co, target)
        target_cpu = tar_this
        if self._get_attr_recur(target, self.labelmapping) is None:
            if self.co.conf['gpu'] is not None:
                tar_this = Variable(torch.FloatTensor([tar_this]).cuda())
            else:
                tar_this = Variable(torch.FloatTensor([tar_this]))

            pred = metaoutput.data.cpu().numpy()[0]
            loss = F.smooth_l1_loss(metaoutput, tar_this)
        else:
            if self.co.conf['gpu'] is not None:
                tar_this = Variable(torch.LongTensor([tar_this]).cuda())
            else:
                tar_this = Variable(torch.LongTensor([tar_this]))

            pred = metaoutput.data.cpu().numpy()[0].argmax()
            loss = F.cross_entropy(metaoutput, tar_this) / bs

        if return_loss:
            return target_cpu, pred, loss.view(1)
        else:
            return target_cpu, pred

    def _correct_counter_recur(self, batch_counter, pred, label, max=False):
        for ky in batch_counter.keys():
            if isinstance(batch_counter[ky], dict):
                self._correct_counter_recur(batch_counter[ky], pred[ky], label[ky], max=max)
            else:
                if max:
                    batch_counter[ky] += 1
                else:
                    batch_counter[ky][label[ky], pred[ky]] += 1

    def _show_count_stats_recur(self, batch_counter, acc_print, sum=0.0, N=0.0, print_stats=False, print_str=''):
        for ky in batch_counter.keys():
            if isinstance(batch_counter[ky], dict):
                sum_, N_ = self._show_count_stats_recur(batch_counter[ky], acc_print, sum, N,
                                                        print_stats, print_str + '/' + ky)
                sum += sum_
                N += N_
            else:
                print_str_ = print_str + ('/' + str(ky))
                if self.co.conf['balanced_eval']:
                    sum__ = 0.0
                    N__ = 0.0
                    a1 = ['relu', 'elu', 'prelu', 'tanh']
                    a2 = ['normal', 'none']
                    a3 = ['max_2', 'none']
                    a4 = [3, 5]
                    a5 = [2, 3, 4]
                    a6 = [2, 3, 4]
                    a7 = ['SGD', 'ADAM', 'RMSprop']
                    a8 = [256, 128, 64]
                    a9 = ['all', 'half_0', 'half_1', 'quarter_0', 'quarter_1', 'quarter_2', 'quarter_3']
                    a10 = ['all', 'half', 'quarter']
                    a11 = [0, 1]
                    a12 = range(14, 22, 1)
                    for iii in range(batch_counter[ky].shape[0]):
                        if print_stats == True:
                            if ky == 'act':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a1[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'drop':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a2[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'pool':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a3[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'ks':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a4[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'n_conv':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a5[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'n_fc':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a6[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'optimiser':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a7[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'batch_size':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a8[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'subset':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a9[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'data_size':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a10[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'ens':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a11[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                            if ky == 'n_param':
                                with st.sidebar:
                                #print("ttt",batch_counter[ky][iii].sum())
                                    with st.expander("Kết quả dự đoán"+" "+ str(ky)+'/'+str(a12[iii])):
                                        st.write(batch_counter[ky][iii].sum())
                        if batch_counter[ky][iii].sum() != 0:
                            sum__ += float(batch_counter[ky][iii][iii]) / batch_counter[ky][iii].sum()
                            N__ += 1
                    acc = sum__ / N__
                    sum += acc
                    rc = 1. / N__
                else:
                    acc = float(np.trace(batch_counter[ky])) / batch_counter[ky].sum()
                    sum += acc
                    rc = float(batch_counter[ky].sum(1).max()) / batch_counter[ky].sum() 
                N += 1
                if print_stats:
                    print("%25s : %2.1f%% (RC %2.1f%%)" % (print_str_, acc * 100, rc * 100))
                    st.write("%25s : %2.1f%% " % (print_str_, acc * 100))
        if print_stats:
            print('                     _____')
        return sum, N

    def _inputnetout_recursive(self, target, query_output):
        metaoutput = {}
        for ky in target.keys():
            if isinstance(target[ky], dict):
                metaoutput[ky] = self._inputnetout_recursive(target[ky], query_output)
            else:
                eyemat = torch.eye(len(target[ky]), 10)
                if self.co.conf['gpu'] is not None:
                    eyemat = eyemat.cuda()
                metaoutput[ky] = torch.mm(Variable(eyemat),
                                          query_output.mean(0).view(-1, 1),
                                          ).view(1, -1)

        return metaoutput

    def train(self, epoch):
        self.epoch = epoch
        which = self.epochmanager[epoch - 1]
        which_init = self.epochmanager_init_m[epoch - 1]
        if which == 'm':
            tl = self.train_loader_metanet
            self.metaoptimizer.zero_grad()
        if which == 'i':
            tl = self.train_loader_input

        s_t = time.time()

        for bi, cos in enumerate(tl):  # for each batch
            pert_batch_dx = np.zeros_like(self.pert)
            batch_loss = 0.0

            batch_counter = initstats_generator(self.target, mat=True)

            for ii in range(tl.batch_size):
                co = cos[ii]

                if ('m' in self.method) and (which_init == 'init'):
                    co.update_token()
                    query_output = self.query_output_fixed['train'][co.token]
                    if self.co.control['setup']['outrep'] == 'argmax':
                        query_output = self._one_hot(query_output)
                    elif 'ranking' in self.co.control['setup']['outrep']:
                        topk = self.co.control['setup']['outrep'].split('-')[1]
                        query_output = self._ranking(query_output, topk)
                    elif 'onlytop' in self.co.control['setup']['outrep']:
                        topk = self.co.control['setup']['outrep'].split('-')[1]
                        query_output = self._onlytop(query_output, topk)

                    if self.co.conf['gpu'] is not None:
                        query_output = torch.from_numpy(query_output).cuda()
                    else:
                        query_output = torch.from_numpy(query_output)

                    query_output = Variable(query_output)
                else:
                    if self.co.conf['gpu'] is not None:
                        pert = torch.from_numpy(self.pert).cuda()
                    else:
                        pert = torch.from_numpy(self.pert)

                    x = Variable(pert, requires_grad=True)
                    query_output = self._compute_queryoutput_ensemble(co, x)

                if 'm' in self.method:
                    metaoutput = self.metanet(query_output)
                else:
                    metaoutput = self._inputnetout_recursive(self.target, query_output)
                loss_ = []
                target_cpu, pred = self._compute_loss_pred_recur(metaoutput, co, tl.batch_size, self.target,
                                                                 return_loss=True, loss=loss_)
                loss = torch.cat(loss_).sum()
                self._correct_counter_recur(batch_counter, pred, target_cpu)

                batch_loss += loss.data.cpu().numpy()

                loss.backward()
                if which == 'i':
                    dx = -x.grad.data
                    pert_batch_dx += dx.cpu().numpy()

            acc_n, acc_d = self._show_count_stats_recur(batch_counter, [], sum=0.0)

            self.curve.curves['train_epoch'].append(epoch)
            self.curve.curves['train_batch_index'].append(bi)
            self.curve.curves['train_batch_loss'].append(batch_loss)
            self.curve.curves['train_batch_avgacc'].append(100 * float(acc_n / acc_d))

            e_t = time.time()
            if (e_t - s_t > 10) or (bi == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAvg acc: {:.1f}%'.format(
                    epoch, bi * tl.batch_size, len(tl.dataset), 100. * bi / len(tl),
                    batch_loss, 100 * float(acc_n / acc_d))
                )
                s_t = e_t

            if which == 'm':
                self.metaoptimizer.step()
                self.metaoptimizer.zero_grad()

            if which == 'i':
                self._add_to_pert(pert_batch_dx)

    def test(self):
        print("Testing..")
        st.write("Testing..")
        tl = self.test_loader
        counter = initstats_generator(self.target, mat=True)

        test_loss = 0.0

        s_t = time.time()
        for bi, cos in enumerate(tl):
            for ii in range(tl.batch_size):
                co = cos[ii]

                if ('m' in self.method) and (self.epochmanager_init_m[self.epoch] == 'init'):
                    co.update_token()
                    query_output = self.query_output_fixed['test'][co.token]
                    if self.co.control['setup']['outrep'] == 'argmax':
                        query_output = self._one_hot(query_output)
                    elif 'ranking' in self.co.control['setup']['outrep']:
                        topk = self.co.control['setup']['outrep'].split('-')[1]
                        query_output = self._ranking(query_output, topk)
                    elif 'onlytop' in self.co.control['setup']['outrep']:
                        topk = self.co.control['setup']['outrep'].split('-')[1]
                        query_output = self._onlytop(query_output, topk)

                    if self.co.conf['gpu'] is not None:
                        query_output = torch.from_numpy(query_output).cuda()
                    else:
                        query_output = torch.from_numpy(query_output)

                    query_output = Variable(query_output, requires_grad=True)
                    # with st.expander("Kết quả dữ liệu cho qua mô hình"):
                    #     st.write(query_output)
                else:
                    if self.co.conf['gpu'] is not None:
                        pert = torch.from_numpy(self.pert).cuda()
                    else:
                        pert = torch.from_numpy(self.pert)

                    x = Variable(pert, requires_grad=True)
                    query_output = self._compute_queryoutput_ensemble(co, x)
                    # with st.expander("Kêt quả dữ liệu cho qua mô hình"):
                    #     st.write(query_output)

                if 'm' in self.method:
                    metaoutput = self.metanet(query_output)
                else:
                    metaoutput = self._inputnetout_recursive(self.target, query_output)

                loss_ = []
                target_cpu, pred = self._compute_loss_pred_recur(metaoutput, co, tl.batch_size, self.target,
                                                                 return_loss=True, loss=loss_)
                loss = torch.cat(loss_).sum()
                self._correct_counter_recur(counter, pred, target_cpu)
                test_loss += loss.data.cpu().numpy()

                e_t = time.time()
                if e_t - s_t > 10:
                    print('Test batch: [{}/{} ({:.0f}%)]'.format(
                        bi * tl.batch_size, len(tl.dataset),
                        100. * bi / len(tl)
                    ))
                    s_t = e_t

        test_loss /= len(tl)

        acc_print = []
        acc_n, acc_d = self._show_count_stats_recur(counter, acc_print, sum=0.0, print_stats=True)
        print('Test loss: {:.6f}, avgacc: {:.2f}%'.format(test_loss, 100 * float(acc_n) / acc_d))
        print(np.array2string(np.array(acc_print), formatter={'float_kind': '{0:.1f}\t'.format}))
        st.write('Test loss: {:.6f}, avgacc: {:.2f}%'.format(test_loss, 100 * float(acc_n) / acc_d))
        #st.write(np.array2string(np.array(acc_print), formatter={'float_kind': '{0:.1f}\t'.format}))

        self.curve.curves['test_epoch'].append(self.epoch)
        self.curve.curves['test_loss'].append(test_loss)
        self.curve.curves['test_avgacc'].append(100 * float(acc_n / acc_d))

        self.accs = acc_print


        return counter

    # def save(self, saveloc):
    #     if 'm' in self.method:
    #         save_to_cache(dict(
    #             pert=mnist_data_transform(self.pert, direction='backward'),
    #             metanet=self.metanet.state_dict(),
    #         ), saveloc)
    #     else:
    #         save_to_cache(dict(
    #             pert=mnist_data_transform(self.pert, direction='backward'),
    #             metanet=self.metanet.state_dict(),
    #         ), saveloc)
    #     self.curve.save(saveloc + '.curve')
    def save(self):
        torch.save(self.metanet,"cache/model.pth")


def main():
    st.set_page_config(page_title='HVT_HVKTQS',page_icon="☀", layout="wide")
    # im = Image.open("Logo_MTA_new.jpg")
    # st.image(im, width=150)
    html_string = "<center><header><h1><font color='green'>REVERSE ENGINEERING NEURAL NETWORK</font></h1></header></center>"
    st.markdown(html_string, unsafe_allow_html=True)
    st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )
    query_params = st.experimental_get_query_params()
    tabs = ["Home", "About Compare"]
    if "tab" in query_params:
        active_tab = query_params["tab"][0]
    else:
        active_tab = "Home"

    if active_tab not in tabs:
        st.experimental_set_query_params(tab="Home")
        active_tab = "Home"

    li_items = "".join(
        f"""
        <li class="nav-item">
            <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}">{t}</a>
        </li>
        """
        for t in tabs
    )
    tabs_html = f"""
        <ul class="nav nav-tabs">
        {li_items}
        </ul>
    """

    st.markdown(tabs_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # if active_tab == "Home":
    #     st.write("Welcome to my lovely page!")
    #     st.write("Feel free to play with this ephemeral slider!")
    #     st.slider(
    #         "Does this get preserved? You bet it doesn't!",
    #         min_value=0,
    #         max_value=100,
    #         value=50,
    #     )
    # elif active_tab == "About":
    #     st.write("This page was created as a hacky demo of tabs")
    # elif active_tab == "Contact":
    #     st.write("If you'd like to contact me, then please don't.")
    # else:
    #     st.error("Something has gone terribly wrong.")
    if active_tab == "Home":
        ip = st.button("Ảnh đầu vào")
        if ip:
            col3, col4 = st.columns([1,3])
            with col3:
                vdt = load_from_cache('cache/mnist_val.pkl', python23_conversion=True)['ip']
                sampler = load_from_cache('cache/mnist_val_queries1000.pkl', python23_conversion=True)[:100]
                pert = vdt[sampler].astype(np.float32)
                with st.expander(""):
                    for iii in range(100):
                        fig = plt.figure()
                        plt.imshow(pert[iii].reshape((28,28)), interpolation = "none",  cmap="gray")
                        st.pyplot(fig)
    
        demo1 = st.button('*Test kennen-o*')
        demo3 = st.button('*Test kennen-io*')
        if demo1:
            # kennen-o approach with 5000 training models and 100 queries with top-5 ranking outputs
            # under the Random (R) split.
            METHOD = 'm'  # Refers to kennen-o
            N_TRAIN = 5000  # Can be chosen in range [100,5000]
            N_EPOCH = 900  # Default number of epochs used in the paper
            N_QUERY = 100  # Can be chosen in range [1,1000]
            OUTPUT = 'ranking-5'  # ranking-k refers to top-k ranking output
            SPLIT = 'rand'
            SPLIT_TR = [1]  # Train on split 1
            SPLIT_TE = [0]  # Test on split 0
            GPU = None  # No GPU
            co = config_metamodel(
            control=dict(
                method=METHOD,
                data=dict(
                    name='dnet10000',
                    subset=N_TRAIN,
                    eval=1000,
                ),
                seed=0,
                i=dict(
                    init='randval',
                    clip=[0, 1],
                    noise='U1',
                    opt=dict(
                        optimiser='SGD',
                        lr=0.1,
                        weight_decay=0.0,
                        batch_size=10,
                    ),
                ),
                m=dict(
                    name='mlp_3_1000',
                    opt=dict(
                        optimiser='SGD',
                        lr=1e-4,
                        weight_decay=0.01,
                        batch_size=100,
                    ),
                ),
                opt=dict(
                    epochs=N_EPOCH,
                    sequence=['m', 200, 50, 50],
                    # sequence=['m', 1, 1, 1],
                ),
                setup=dict(
                    nquery=N_QUERY,
                    qseed=0,
                    target='all',
                    outrep=OUTPUT,
                    split=SPLIT,
                    splitidtr=SPLIT_TR,
                    splitidte=SPLIT_TE,
                ),
            ),
            conf=dict(
                exp_phase='mnist_metamodel',
                balanced_eval=True,
                test_batch_size=10,
                test_epoch=1,
                save=False,
                overridecache=True,
                mode='train',
                gpu=GPU,
            )
        )
            st.snow()
            col1, col2 = st.columns([1,3])
            with col1:
                net = blackBoxRevealer(co)

                net.test() 
            with col2:
                # st2 = "<table> <tr> <th>Company</th> <th>Contact</th> <th>Country</th> </tr> <tr> <td>Alfreds Futterkiste</td> <td>Maria Anders</td> <td>Germany</td> </tr> <tr> <td>Centro comercial Moctezuma</td> <td>Francisco Chang</td> <td>Mexico</td> </tr> </table>"
                # st.markdown(st2, unsafe_allow_html=True)
                html_string = "<center><header><h1><font color='blue'>BIỂU ĐỒ KẾT QUẢ</font></h1></header></center>"
                st.markdown(html_string, unsafe_allow_html=True)
                # im = Image.open(r"black.png") 
                # fig = plt.figure()
                # plt.imshow(im, interpolation = "none",  cmap="gray")
                # st.pyplot(fig)
                data1 = {'/net/act' :63.2, '/nte/drop' :93.9, '/net/pool' :91.6, '/net/ks' :81.3, '/net/n_conv' :66.7, '/net/n_fc' :68.9, '/opt/optimiser' :65.9, '/opt/batch_size' :49.9, '/data/subset' :84.6, '/etc/data_size' :72.2, '/etc/ens' :72.2, '/etc/n_param' :46.6}
                courses = list(data1.keys())
                values = list(data1.values())
                fig = plt.figure(figsize = (25,10))
                plt.bar(courses,values)
                plt.xlabel("THÀNH PHẦN TRÍCH XUẤT")
                plt.ylabel("TỶ LỆ TRÍCH XUẤT")
                #plt.title("BIỂU ĐỒ KẾT QUẢ")
                st.pyplot(fig)
    
        if demo3:
            # kennen-io approach with 100 training models and 100 queries with score outputs
            # under the Extrapolation (E) split, with splitting attribute {#conv,#fc}.
            METHOD = 'mi'  # Refers to kennen-io
            N_TRAIN = 100
            N_EPOCH = 400  # Default number of epochs for kennen-io
            N_QUERY = 100
            OUTPUT = 'score'
            SPLIT = 'ex^net/n_conv^net/n_fc'  # Possible to set multiple splitting attributes separated via '^'
            SPLIT_TR = [0, 1]  # Train on #conv=#fc=2 or 3
            SPLIT_TE = [2]  # Test on #conf=#fc=4
            GPU = None  # GPU ID
            co = config_metamodel(
            control=dict(
                method=METHOD,
                data=dict(
                    name='dnet10000',
                    subset=N_TRAIN,
                    eval=1000,
                ),
                seed=0,
                i=dict(
                    init='randval',
                    clip=[0, 1],
                    noise='U1',
                    opt=dict(
                        optimiser='SGD',
                        lr=0.1,
                        weight_decay=0.0,
                        batch_size=10,
                    ),
                ),
                m=dict(
                    name='mlp_3_1000',
                    opt=dict(
                        optimiser='SGD',
                        lr=1e-4,
                        weight_decay=0.01,
                        batch_size=100,
                    ),
                ),
                opt=dict(
                    epochs=N_EPOCH,
                    sequence=['m', 200, 50, 50],
                    # sequence=['m', 1, 1, 1],
                ),
                setup=dict(
                    nquery=N_QUERY,
                    qseed=0,
                    target='all',
                    outrep=OUTPUT,
                    split=SPLIT,
                    splitidtr=SPLIT_TR,
                    splitidte=SPLIT_TE,
                ),
            ),
            conf=dict(
                exp_phase='mnist_metamodel',
                balanced_eval=True,
                test_batch_size=10,
                test_epoch=1,
                save=False,
                overridecache=True,
                mode='train',
                gpu=GPU,
            )
        )
            st.snow()
            col1, col2 = st.columns([1,3])
            with col1:
                net = blackBoxRevealer(co)

                net.test() 
            with col2:
                data1 = {'/net/act' :26.7, '/nte/drop' :64.5, '/net/pool' :74.0, '/net/ks' :55.7, '/net/n_conv' :59.0, '/net/n_fc' :41.6, '/opt/optimiser' :32.9, '/opt/batch_size' :34.0, '/data/subset' :44.1, '/etc/data_size' :49.1, '/etc/ens' :49.8, '/etc/n_param' :28.9}
                #data1 = {'/net/act' :63.0, '/nte/drop' :93.4, '/net/pool' :90.5, '/net/ks' :81.9, '/net/n_conv' :64.2, '/net/n_fc' :68.5, '/opt/optimiser' :63.9, '/opt/batch_size' :48.3, '/data/subset' :86.1, '/etc/data_size' :72.2, '/etc/ens' :72.2, '/etc/n_param' :44.2}
                courses = list(data1.keys())
                values = list(data1.values())
                fig = plt.figure(figsize = (25,10))
                plt.bar(courses,values)
                plt.xlabel("THÀNH PHẦN TRÍCH XUẤT")
                plt.ylabel("TỶ LỆ TRÍCH XUẤT")
                plt.title("BIỂU ĐỒ KẾT QUẢ")
                st.pyplot(fig)
    elif active_tab == "About Compare":
        bt = st.button('Kennen-o vs Random')
        bt1 = st.button('Kennen-io vs Random')
        if bt:
            st5 = "<table> <caption><h2>_________________THỐNG KÊ KẾT QUẢ DỰ ĐOÁN__________________</h2></caption> <tr><th>**********</th><th>/net/act:</th><th>/net/drop:</th><th>/net/pool:</th><th>/net/ks:</th><th>/net/n_conv:</th><th>/net/n_fc:</th><th>/opt/optimiser:</th><th>/opt/batch_size:</th><th>/data/subset:</th><th>/etc/data_size</th><th>/etc/ens:</th><th>/etc/n_param:</th><th>Avgacc:</th></tr> <tr><td>Kennen-o</td><td>62.2%</td><td>93.9%</td><td>91.6%</td><td>81.3%</td><td>66.7%</td><td>68.9%</td><td>65.9%</td><td>49.9%</td><td>84.6%</td><td>72.2%</td><td>72.2%</td><td>46.6%</td><td>74.56%</td></tr><tr><td>Random</td><td>25.0%%</td><td>50.0%</td><td>50.0%</td><td>50.0%</td><td>33.3%</td><td>33.3%</td><td>33.3%</td><td>33.3%</td><td>14.3%</td><td>33.3%</td><td>50.0%</td><td>14.3%</td><td>34.9%</td></tr> </table>"
            st.markdown(st5, unsafe_allow_html=True)
            col5, col6 = st.columns([2,2])
            with col5:
                
                ###
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                html_string = "<center><header><h1><font color='blue'>BIỂU ĐỒ KẾT QUẢ KENNEN-O (Tỷ lệ: 100%)</font></h1></header></center>"
                st.markdown(html_string, unsafe_allow_html=True)
                # im = Image.open(r"black.png") 
                # fig = plt.figure()
                # plt.imshow(im, interpolation = "none",  cmap="gray")
                # st.pyplot(fig)
                data1 = {'/net/act' :63.2, '/nte/drop' :93.9, '/net/pool' :91.6, '/net/ks' :81.3, '/net/n_conv' :66.7, '/net/n_fc' :68.9, '/opt/optimiser' :65.9, '/opt/batch_size' :49.9, '/data/subset' :84.6, '/etc/data_size' :72.2, '/etc/ens' :72.2, '/etc/n_param' :46.6}
                courses = list(data1.keys())
                values = list(data1.values())
                fig = plt.figure(figsize = (25,10))
                plt.bar(courses,values)
                plt.xlabel("THÀNH PHẦN TRÍCH XUẤT")
                plt.ylabel("TỶ LỆ TRÍCH XUẤT")
                #plt.title("BIỂU ĐỒ KẾT QUẢ")
                st.pyplot(fig)
            ####
            with col6:
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                html_string = "<center><header><h1><font color='blue'>BIỂU ĐỒ KẾT QUẢ RANDOM (Tỷ lệ: 50%)</font></h1></header></center>"
                st.markdown(html_string, unsafe_allow_html=True)
                ####
                data1 = {'/net/act' :25.0, '/nte/drop' :50.0, '/net/pool' :50.0, '/net/ks' :50.0, '/net/n_conv' :33.3, '/net/n_fc' :33.3, '/opt/optimiser' :33.3, '/opt/batch_size' :33.3, '/data/subset' :14.3, '/etc/data_size' :33.3, '/etc/ens' :50.0, '/etc/n_param' :14.3}
                courses = list(data1.keys())
                values = list(data1.values())
                fig = plt.figure(figsize = (25,10))
                plt.bar(courses,values)
                plt.xlabel("THÀNH PHẦN TRÍCH XUẤT")
                plt.ylabel("TỶ LỆ TRÍCH XUẤT")
                #plt.title("BIỂU ĐỒ KẾT QUẢ")
                st.pyplot(fig)
        if bt1:
            st5 = "<table> <caption><h2>_________________THỐNG KÊ KẾT QUẢ DỰ ĐOÁN__________________</h2></caption> <tr><th>**********</th><th>/net/act:</th><th>/net/drop:</th><th>/net/pool:</th><th>/net/ks:</th><th>/net/n_conv:</th><th>/net/n_fc:</th><th>/opt/optimiser:</th><th>/opt/batch_size:</th><th>/data/subset:</th><th>/etc/data_size</th><th>/etc/ens:</th><th>/etc/n_param:</th><th>Avgacc:</th></tr> <tr><td>Kennen-io</td><td>26.7%</td><td>64.5%</td><td>74.0%</td><td>55.7%</td><td>59.0%</td><td>41.6%</td><td>32.9%</td><td>34.0%</td><td>44.1%</td><td>49.1%</td><td>49.8%</td><td>28.9%</td><td>50.10%</td></tr><tr><td>Random</td><td>25.0%%</td><td>50.0%</td><td>50.0%</td><td>50.0%</td><td>33.3%</td><td>33.3%</td><td>33.3%</td><td>33.3%</td><td>14.3%</td><td>33.3%</td><td>50.0%</td><td>14.3%</td><td>34.9%</td></tr> </table>"
            st.markdown(st5, unsafe_allow_html=True)
            col7, col8 = st.columns([2,2])
            with col7:
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                html_string = "<center><header><h1><font color='blue'>BIỂU ĐỒ KẾT QUẢ KENNEN-IO (Tỷ lệ: 100%)</font></h1></header></center>"
                st.markdown(html_string, unsafe_allow_html=True)
                # im = Image.open(r"black.png") 
                # fig = plt.figure()
                # plt.imshow(im, interpolation = "none",  cmap="gray")
                # st.pyplot(fig)
                data1 = {'/net/act' :26.7, '/nte/drop' :64.5, '/net/pool' :74.0, '/net/ks' :55.7, '/net/n_conv' :59.0, '/net/n_fc' :41.6, '/opt/optimiser' :32.9, '/opt/batch_size' :34.0, '/data/subset' :44.1, '/etc/data_size' :49.1, '/etc/ens' :49.8, '/etc/n_param' :28.9}
                courses = list(data1.keys())
                values = list(data1.values())
                fig = plt.figure(figsize = (25,10))
                plt.bar(courses,values)
                plt.xlabel("THÀNH PHẦN TRÍCH XUẤT")
                plt.ylabel("TỶ LỆ TRÍCH XUẤT")
                #plt.title("BIỂU ĐỒ KẾT QUẢ")
                st.pyplot(fig)
            with col8:
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                html_string = "<center><header><h1><font color='blue'>BIỂU ĐỒ KẾT QUẢ RANDOM (Tỷ lệ: 50%)</font></h1></header></center>"
                st.markdown(html_string, unsafe_allow_html=True)
                ####
                data1 = {'/net/act' :25.0, '/nte/drop' :50.0, '/net/pool' :50.0, '/net/ks' :50.0, '/net/n_conv' :33.3, '/net/n_fc' :33.3, '/opt/optimiser' :33.3, '/opt/batch_size' :33.3, '/data/subset' :14.3, '/etc/data_size' :33.3, '/etc/ens' :50.0, '/etc/n_param' :14.3}
                courses = list(data1.keys())
                values = list(data1.values())
                fig = plt.figure(figsize = (25,10))
                plt.bar(courses,values)
                plt.xlabel("THÀNH PHẦN TRÍCH XUẤT")
                plt.ylabel("TỶ LỆ TRÍCH XUẤT")
                #plt.title("BIỂU ĐỒ KẾT QUẢ")
                st.pyplot(fig)
   
if __name__ == '__main__':
    main()