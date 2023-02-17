"""
author: nabin

"""


import numpy as np
import os
import cmath
import math
import random
import sys
import pickle

random.seed(10)

try:
    source_base = open("ua.base", "r")
    #source_test = open("ua.test", "r")
except IOError:
    print(
        "The demo require input file ua.base, ua.test from ml-100K in current folder.\nPlease get them from www.grouplens.org")
    exit()
if len(sys.argv) < 3:
    print(" Error: Usage: <input> \n", "Example: program.py ua.base ua.test")
    exit()
print("Predicting using Movie lens data")
print("All data loaded in.\nLine shuffle start.")
base_data = [(random.random(), line) for line in source_base]
print("Line Shuffle End :Data Shuffled")
base_data.sort()
with open("ua.base.shuffle", "+w") as target_base:
    for _, line in base_data:
        target_base.write(line)

source_base.close()
target_base.close()



def group_shuffle(fin, fout):
    d = {}
    k = open(fout, "+w")
    with open(fin) as file:
        for line in file:
            line_val = line.split()
            key = line_val[0]
            if key in d:
                val = d[key]
                val.append(line)
            else:
                d[key] = [line]
    value = list(d.values())
    random.seed(4)
    random.shuffle(value)
    for data in value:
        for datas in data:
            k.write('{}'.format(datas))

    file.close()


def mkfeature_basic(fin, fout):
    fi = open(fin, "r")
    fo = open(fout, "+w")

    for line in fi:
        arr = line.split("\t")
        uid = int(arr[0].strip())
        iid = int(arr[1].strip())
        score = float(arr[2].strip())
        fo.write("%d\t0\t1\t1\t" % score)
        fo.write("%d:1 %d:1\n" % (uid - 1, iid - 1))
    fi.close()
    fo.close()


# user implicit feedback
def userfeedback(fname):
    fi = open(fname, 'r')
    feedback = {}
    for line in fi:
        attr = line.strip().split()
        uid = int(attr[0]) - 1
        iid = int(attr[1]) - 1
        if uid in feedback:
            feedback[uid].append(iid)
        else:
            feedback[uid] = [iid]
    fi.close()
    return feedback


# group num and order of the grouped training data
def usergroup(fname):
    fi = open(fname, 'r')
    userorder = []
    groupnum = {}
    lastuid = -1
    for line in fi:
        attr = line.strip().split('\t')
    uid = int(attr[0]) - 1
    if uid in groupnum:
        groupnum[uid] += 1
    else:
        groupnum[uid] = 1
    if uid != lastuid:
        userorder.append(uid)
    lastuid = uid
    fi.close()
    return userorder, groupnum


# make implict feedback feature, one line for a user, wihch is in the order of the grouped training data
# the output format:rate \t number of user group \t number of user implicit feedback \t fid1:fvalue1, fid2:fvalue2 ... \n
def mkfeature(fout, userorder, groupnum, feedback):
    fo = open(fout, 'w')
    for uid in userorder:
        gnum = groupnum[uid]
        fnum = len(feedback[uid])
        fo.write('%d\t%d\t' % (gnum, fnum))
        for i in feedback[uid]:
            fo.write('%d:%.6f ' % (i, pow(fnum, -0.5)))
        fo.write('\n')


class config:

    def __init__(self):
        self.cfg = []
        self.tmp2 = []
        self.tmp1 = []
        self.name_config = "config.conf"
        with open(self.name_config) as file:
            for line in file:
                line_value = line.split()
                self.tmp2.append(line_value)

        for i in range(0, len(self.tmp2)):
            if len(self.tmp2[i]) < 4 and len(self.tmp2[i]) != 0:
                self.tmp1.append(self.tmp2[i])

        for i in range(0, len(self.tmp1)):
            name = self.tmp1[i][0]
            value = self.tmp1[i][2]
            self.cfg.append((name, value))
        file.close()


RANDOM_ORDER_FORMAT = 0
USER_GROUP_FORMAT = 1
AUTO_DETECT = 2

LINEAR = 'LINEAR'
SIGMOID_L2 = "SIGMOID_L2"
SIGMOID_LIKELIHOOD = "SIGMOID_LIKELIHOOD"
SIGMOID_RANK = "SIGMOID_RANK"
HINGE_SMOOTH = "HINGE_SMOOTH"
HINGE_L2 = "HINGE_L2"
SIGMOID_QSGRAD = "SIGMOID_QSGRAD"


def active_type(name, value):
    active = None
    if name == "active_type":
        if value == '0':
            active = LINEAR
        elif value == '1':
            active = SIGMOID_L2
        elif value == '2':
            active = SIGMOID_LIKELIHOOD
        elif value == '3':
            active = SIGMOID_RANK
        elif value == '5':
            active = HINGE_SMOOTH
        elif value == '6':
            actve = HINGE_L2
        elif value == '7':
            active = SIGMOID_QSGRAD
    return active


def map_active(total, activation):
    if activation == LINEAR:
        return total
    elif activation == SIGMOID_L2:
        return None
    elif activation == SIGMOID_LIKELIHOOD:
        return 1.0 / (1.0 + math.exp(-total))
    elif activation == SIGMOID_RANK:
        return total
    elif activation == HINGE_SMOOTH:
        return total
    elif activation == HINGE_L2:
        return total
    elif activation == SIGMOID_QSGRAD:
        return total
    else:
        print("Unknown Active Type")
        return 0.0


def sqr(z):
    return z * z


def smooth_hinge_grad(z):
    if z > 1.0:
        return 0.0
    elif z < 0.0:
        return 1.0
    return 1.0 - z


def smooth_hinge_loss(z):
    if z > 1.0:
        return 0.0
    elif z > 0.0:
        return 0.5 * ((1.0 - z) * (1.0 - z))


def cal_grad(r, pred, activation):
    if activation == LINEAR:
        return r - pred
    elif activation == SIGMOID_L2:
        return (r - pred) * pred * (1 - pred)
    elif activation == SIGMOID_LIKELIHOOD:
        return r - pred
    elif activation == SIGMOID_QSGRAD:
        return None
    elif activation == SIGMOID_RANK:
        return r - 1.0 / (1.0 + math.exp(-pred))
    elif activation == HINGE_SMOOTH:
        return smooth_hinge_grad((pred - 0.5) if (r > 0.5) else (- smooth_hinge_grad(0.5 - pred)))
    elif activation == HINGE_L2:
        if r > 0.5:
            if pred > 1.0:
                return 0.0
            else:
                return r - pred
        elif pred < 0.0:
            return 0.0
        else:
            return r - pred
    else:
        print("Unknown Active Type")
        return 0.0


def cal_loss(r, pred, activation):
    config_file = config()
    for name, value in config_file.cfg:
        if name == active_type:
            active = active_type(name, value)
    if active == LINEAR:
        return None
    elif active == SIGMOID_L2:
        return 0.5 * ((r - pred) * (r - pred))
    elif active == SIGMOID_QSGRAD:
        return None
    elif active == SIGMOID_RANK:
        pred = (1.0 / (1.0 + math.exp(- pred)))
    elif active == SIGMOID_LIKELIHOOD:
        return - r * math.log(pred) - (1.0 - r) * math.log(pred)
    elif active == HINGE_SMOOTH:
        pred -= 0.5
        if r > 0.5:
            return smooth_hinge_loss(pred)
        else:
            return -smooth_hinge_loss(-pred)
    elif active == HINGE_L2:
        if r > 0.5:
            if pred > 1.0:
                pred = 1.0
                return 0.5 * ((1.0 - pred) * (1.0 - pred))
            elif pred < 0.0:
                pred = 0.0
                return 0.5 * (pred * pred)
    else:
        print("Unknown Active Type ")
        return 0.0


def cal_sgrad(r, pred, activation):
    if activation == LINEAR:
        return -1.0
    elif activation == SIGMOID_LIKELIHOOD:
        return - pred * (1.0 - pred)
    elif activation == SIGMOID_RANK:
        pred = 1.0 / (1.0 + math.exp(-pred))
        return -pred * (1.0 - pred)
    elif activation == HINGE_SMOOTH:
        return None
    elif activation == HINGE_L2:
        return -1.0
    elif activation == SIGMOID_QSGRAD:
        return -0.25
    else:
        print("Unknown Second order Gradient for active type")
        return 0.0


def calc_base_score(base_score, activation):
    if activation == '1':
        return None
    elif activation == '6':
        return None
    elif activation == '5':
        return base_score
    elif activation == '0':
        return None
    elif activation == '2':
        return None
    elif activation == '3':
        return None
    elif activation == '7':
        assert '0.0' < base_score < '1.0', "Sigmoid range Constrain"
        return -math.log(1.0 / int(base_score) - 1.0)
    else:
        print(" Unknown Active Type")
        return 0.0


class SVDTypeParam:

    def __init__(self):
        # try to decide the format.
        self.format_type = AUTO_DETECT
        self.active_type = self.extend_type = self.variant_type = 0
        self.config = config()
        for name, value in self.config.cfg:
            self.set_param_type(name, value)

    def set_param_type(self, name, value):
        if name == "model_type":
            self.model_type = value
        if name == "format_type":
            self.format_type = value
        if name == "active_type":
            self.active_type = value
        if name == "extend_type":
            self.extend_type = value
        if name == "variant_type":
            self.variant_type = value

    def decide_format(self):
        if self.format_type != AUTO_DETECT:
            return None
        else:
            self.format_type = RANDOM_ORDER_FORMAT if self.extend_type == 0 else USER_GROUP_FORMAT
            return self.format_type


# brief default training parameters for SVDFeature
class SVDTrainParam:

    def __init__(self):
        self.learning_rate = 0.01
        self.reg_method = 0
        self.wd_user = 0.0
        self.wd_user_bias = self.wd_item_bias = 0.0
        self.num_regfree_global = 0
        self.reg_global = 0
        self.wd_global = 0.0
        self.decay_learning_rate = 0
        self.decay_rate = 1.0
        self.min_learning_rate = 0.0
        self.scale_lr_ufeedback = 1.0
        self.wd_ufeedback = self.wd_ufeedback_user = self.wd_ufeedback_bias = 0
        self.config = config()
        for name, value in self.config.cfg:
            self.set_param_train(name, value)

    def set_param_train(self, name, value):
        if name == "learning_rate":
            self.learning_rate = value
        elif name == "wd_user":
            self.wd_user = value
        elif name == "wd_item":
            self.wd_item = value
        elif name == "wd_uiset":
            self.wd_user = self.wd_item = value
        elif name == "wd_user_bias":
            self.wd_user_bias = value
        elif name == "wd_item_bias":
            self.wd_item_bias = value
        elif name == "wd_uiset_bias":
            self.wd_user_bias = self.wd_item_bias = value
        elif name == "wd_global":
            self.wd_global = value
        elif name == "reg_method":
            self.reg_method = value
        elif name == "reg_global":
            self.reg_global = value
        elif name == "num_regfree_global":
            self.num_regfree_global = value
        elif name == "decay_learning_rate":
            self.decay_learning_rate = value
        elif name == "min_learning_rate":
            self.min_learning_rate = value
        elif name == "decay_rate":
            self.decay_rate = value
        elif name == "scale_lr_ufeedback":
            self.scale_lr_ufeedback = value
        elif name == "wd_ufeedback":
            self.wd_ufeedback = value
        elif name == "wd_ufeedback_bias":
            self.wd_ufeedback_bias = value


class SVDModelParam:

    def __init__(self):
        self.num_user = self.num_item = self.num_global = self.num_factor = 0
        self.u_init_sigma = self.i_init_sigma = float(0.01)  # std variance for user and item factor
        self.no_user_bias = float(0.0)
        # global mean of prediction
        self.base_score = float(0.5)
        self.num_ufeedback = 0
        self.ufeedback_init_sigma = float(0.0)
        self.num_randinit_ufactor = self.num_randinit_ifactor = 0
        self.common_latent_space = 0
        self.user_nonnegative = 0
        self.item_nonnegative = 0
        self.common_feedback_space = 0
        self.extend_flag = 0
        self.config = config()
        for name, value in self.config.cfg:
            self.set_param(name, value)

    def set_param(self, name, value):
        if name == "num_user":
            self.num_user = value
        elif name == "num_item":
            self.num_item = value
        elif name == "num_uiset":
            self.num_user = self.num_item = value
        elif name == "num_global":
            self.num_global = value
        elif name == "num_factor":
            self.num_factor = value
        elif name == "u_init_sigma":
            self.u_init_sigma = value
        elif name == "i_init_sigma":
            self.i_init_sigma = value
        elif name == "ui_init_sigma":
            self.u_init_sigma = self.i_init_sigma = value
        elif name == "base_score":
            self.base_score = value
        elif name == "no_user_bias":
            self.no_user_bias = value
        elif name == "num_ufeedback":
            self.num_ufeedback = value
        elif name == "num_randinit_ufactor":
            self.num_randinit_ufactor = value
        elif name == "num_randinit_ifactor":
            self.num_randinit_ifactor = value
        elif name == "num_randinit_uifactor":
            self.num_randinit_ufactor = self.num_randinit_ifactor = value
        elif name == "ufeedback_init_sigma":
            self.ufeedback_init_sigma = value
        elif name == "common_latent_space":
            self.common_latent_space = value
        elif name == "common_feedback_space":
            self.common_feedback_space = value
        elif name == "user_nonnegative":
            self.user_nonnegative = value
        elif name == "item_nonnegative":
            self.item_nonnegative = value


class SVDModel:

    def __init__(self):
        self.param = SVDModelParam()
        self.mtype = SVDTypeParam()
        self.feature_csr = SVDFeatureCSR()
        self.alloc_space()
        self.rand_init()

    def alloc_space(self):
        self.ustart = self.param.num_ufeedback if self.param.common_feedback_space == 0 and self.mtype.format_type == USER_GROUP_FORMAT else 0
        if self.param.common_latent_space == 0:
            self.ui_bias = np.zeros(self.ustart + int(self.param.num_user) + int(self.param.num_item))
            self.W_uiset = np.zeros(
                (self.ustart + int(self.param.num_user) + int(self.param.num_item), int(self.param.num_factor)))
        else:
            assert int(self.param.num_user) == int(
                self.param.num_item), "num user and num item must be same to use common latent space"
            assert self.param.common_feedback_space is not 0, " common latent space must enforce common feedback space"
            self.ui_bias = np.zeros(int(self.param.num_user + self.param.num_item))
            self.W_uiset = np.zeros((int(self.param.num_item), int(self.param.num_factor)))

        if self.param.common_latent_space == 0:
            self.u_bias = self.ui_bias[self.ustart:int(self.param.num_user)]
            self.W_user = self.W_uiset[self.ustart:int(self.param.num_user), 0:int(self.param.num_factor)]
            self.i_bias = np.zeros((int(self.param.num_item)))
            self.W_item = self.W_uiset[int(self.param.num_user) + self.ustart: -1,
                          0:int(self.param.num_factor)]

        else:
            self.W_user = self.W_uiset[self.ustart:int(self.param.num_user), 0:int(self.param.num_factor)]
            self.u_bias = self.ui_bias[self.ustart:int(self.param.num_user)]
            self.W_item = self.W_uiset[int(self.param.num_user):-1, 0:int(self.param.num_factor)]
            self.i_bias = np.zeros((int(self.param.num_item)))

        self.g_bias = np.zeros(int(self.param.num_global))

        if self.mtype.format_type == USER_GROUP_FORMAT:
            if self.param.common_feedback_space == 0:
                self.ufeedback_bias = self.ui_bias[0:int(self.param.num_ufeedback)]
                self.W_ufeedback = self.W_uiset[0:int(self.param.num_ufeedback), 0:int(self.param.num_factor)]
            else:
                self.ufeedback_bias = self.u_bias
                self.W_ufeedback = self.W_user

    def save_to_file(self, filename):
        file = open(filename.name, "wb")
        if self.param.common_latent_space == 0:
            pickle.dump(self.u_bias, file)
            pickle.dump(self.W_user, file)
            pickle.dump(self.i_bias, file)
            pickle.dump(self.W_item, file)
        else:
            pickle.dump(self.ui_bias, file)
            pickle.dump(self.W_uiset, file)

        pickle.dump(self.g_bias, file)

        if self.mtype.format_type == USER_GROUP_FORMAT:
            if self.param.common_feedback_space == 0:
                pickle.dump(self.ufeedback_bias, file)
                pickle.dump(self.W_ufeedback, file)

    def load_from_file(self, filename):
        if type(filename) == str:
            model_file = filename
        else:
            model_file = filename.name
        if os.stat(model_file).st_size == 0:
            print("Error loading CF SVD model")
            exit(1)
        try:
            file = open(model_file, "rb")
        except IOError:
            print("There is no model file")
        if self.param.common_latent_space == 0:
            self.u_bias = pickle.load(file)
            assert len(self.u_bias) > 0, "load from file"
            self.W_user = pickle.load(file)
            self.i_bias = pickle.load(file)
            self.W_item = pickle.load(file)
        else:
            self.ui_bias = pickle.load(file)
            self.W_uiset = pickle.load(file)
        self.g_bias = pickle.load(file)
        if self.mtype.format_type == USER_GROUP_FORMAT:
            if self.param.common_feedback_space == 0:
                self.ufeedback_bias = pickle.load(file)
                self.W_ufeedback = pickle.load(file)


    def sample_normal(self):
        while True:
            x = 2 * random.random() - 1.0
            y = 2 * random.random() - 1.0
            s = x * x + y * y
            if s < 1.0 or s != 0:
                break
        return x * (cmath.sqrt((-2.0) * (math.log(s) / s)))

    def rand_init(self):
        self.param.base_score = calc_base_score(self.param.base_score, self.mtype.active_type)
        if int(self.param.num_randinit_ufactor) is not 0:
            self.W_uinit = np.copy(self.W_user[0:self.param.num_randinit_ufactor, 0:])
        else:
            self.W_uinit = np.copy(self.W_user)

        for i in range(len(self.W_uinit)):
            self.W_uinit[i] = self.sample_normal() * self.param.u_init_sigma

        if self.param.user_nonnegative:
            for y in range(len(self.W_user)):
                for x in range(len(self.W_user[0])):
                    self.W_user[y][x] = abs(self.W_user[y][x])

        if self.param.common_latent_space == 0:
            if self.param.num_randinit_ifactor is not 0:
                self.W_iinit = np.copy(self.W_item[0:self.param.num_randinit_ifactor, 0:])
            else:
                self.W_iinit = np.copy(self.W_item)

        for i in range(len(self.W_iinit)):
            self.W_iinit[i] = self.sample_normal() * self.param.i_init_sigma

        if self.param.item_nonnegative:
            for y in range(len(self.W_iinit)):
                for x in range(len(self.W_iinit[0])):
                    self.W_iinit[y][x] = abs(self.W_iinit[y][x])

        if self.mtype.format_type == USER_GROUP_FORMAT:
            pass


class parameterset:
    def __init__(self, prefixA, prefixB):
        self.prefixA = prefixA
        self.prefixB = prefixB
        self.prefixA_lenA = len(self.prefixA)
        self.prefixB_lenB = len(self.prefixB)
        # weight decay
        self.__wd = []
        # bound for index
        self.__bound = []

    def set_param(self, name, value):
        if name.startswith(self.prefixA):
            name = name[self.prefixA_lenA:]
        elif name.startswith(self.prefixB):
            name = name[self.prefixB_lenB:]
        else:
            return
        if name == "bound":
            bd = int(value)
            assert bd > 0, "can't give 0 as bound"
            assert len(self.__bound) == 0 or self.__bound[-1] < bd, "bound must be given in order"
            assert len(self.__bound) + 1 == len(self.__wd), "must specify wd in each range"
            self.__bound.append(bd - 1)

        if name == "wd":
            assert (len(self.__wd) == len(self.__bound)), "setting must be exact"
            self.__wd.append(float(value))

    def get_wd(self, gid, wd_default):
        if len(self.__bound) == 0:
            return wd_default
        idx = self.__bound.index(gid)
        assert idx < len(self.__bound), "bound set err"
        return self.__wd[idx]


class Elem:
    def __init__(self):
        # result label , rate or {0,1} for classification.
        self.label = float(0)
        # number of non-zero global feature
        self.num_global = 0
        # number of non-zero user feature
        self.num_ufactor = 0
        # number of non-zero item feature
        self.num_ifactor = 0
        # array of global feature index
        self.index_global = []
        # array of user feature index
        self.index_ufactor = []
        # array of item feature index
        self.index_ifactor = []
        # array of global feature value
        self.value_global = []
        # array of user feature value
        self.value_ufactor = []
        # array of item feature value
        self.value_ifactor = []
        # array of feedback
        self.feedback_id = []
        self.feedback_value = []


class SVDFeatureCSR:

    def load_from_file(self, filename):
        with open(filename, "r") as data_file:
            for line in data_file:
                elem = Elem()
                line_values = line.split()
                elem.label = [int(line_values[0])]
                elem.num_global = [int(line_values[1])]
                elem.num_ufactor = [int(line_values[2])]
                elem.num_ifactor = [int(line_values[3])]
                # index_global = [(line_values[4])]
                ufactor = [(line_values[4])]
                ifactor = [(line_values[5])]
                for i in ifactor:
                    value_i = i.split(":")
                    elem.index_ifactor.append(value_i[0])
                    elem.value_ifactor.append(value_i[1])

                for i in ufactor:
                    value_u = i.split(":")
                    elem.index_ufactor.append(value_u[0])
                    elem.value_ufactor.append(value_u[1])

                self.elems.append(elem)

        with open("ua.base.feedbackfeature", "r") as file:
            for line in file:
                line_values = line.split()
                n = int(line_values[1])
                elem.num_feedback = n
                for i in range(2, n):
                    feedback_val = line_values[i]
                    f = feedback_val.split(":")
                    elem.feedback_id.append(f[0])
                    elem.feedback_value.append(f[1])

            self.elems.append(elem)

        file.close()
        data_file.close()

    def __init__(self):
        self.elems = []
        self.load_from_file(filename)
        self.num_ufeedback = 0
        self.name_data = None
        self.scale_score = float(0.0)
        self.config = config()
        self.save_to_file()

        for name, value in self.config.cfg:
            self.set_param(name, value)

    def set_param(self, name, value):
        if name == "scale_score":
            self.scale_score = value
        if name == "data_in":
            self.name_data = value

    def save_to_file(self):
        pass


class SVDFeature:
    def __init__(self):
        self.config = config()
        self.model = SVDTypeParam()
        for name, value in self.config.cfg:
            if name == "active_type":
                active_type(name, value)
                break
        self.active_type = active_type(name, value)
        self.feature = SVDFeatureCSR()
        self._model = SVDModel()
        self.param = SVDModelParam()
        self._param = SVDTrainParam()
        self.__round_counter = 0
        self.__name_feat_user = None
        self.__name_feat_item = None
        self._sample_counter = 0
        # data structure for detail weight decay tuning
        self.u_param = parameterset("up:", "uip:")
        self.i_param = parameterset("ip:", "uip:")
        self.g_param = parameterset("gp:", "gp")
        self._feat_user = []
        self._feat_item = []
        self._init_end = 0

    def set_param(self, name, value):
        if name == "feature_user":
            self.__name_feat_user = value
        if name == "feature_item":
            self.__name_feat_item = value
        self.u_param.set_param(name, value)
        self.i_param.set_param(name, value)
        self.g_param.set_param(name, value)

    def load_model(self, filename):
        self._model.load_from_file(filename)

    def save_model(self, filename):
        self._model.save_to_file(filename)

    def init_model(self):
        self._model.alloc_space()
        self._model.rand_init()

    def init_trainer(self):
        if self.__name_feat_user is not None:
            self._feat_user.append(self.__name_feat_user)
        if self.__name_feat_item is not None:
            self._feat_item.append(self.__name_feat_item)
        self._tmp_ufactor = self._model.W_user[0]
        self._tmp_ifactor = self._model.W_item[0]
        self._sample_counter = 0
        if self._param.reg_global >= 4:
            self.ref_global = []
        if self._param.reg_method >= 4:
            self.ref_user = []
            if self._model.param.common_latent_space == 0:
                self.ref_item = []
            else:
                self.ref_item = self.ref_user
        self._init_end = 1

    def _reg_L1(self, w, wd):
        if w > wd:
            w -= wd
        else:
            if w < -wd:
                w += wd
            else:
                w = float(0.0)
        return w

    def _project(self, w, B):
        total = float(np.dot(w, w))
        if total > B:
            for i in range(len(w)):
                w[i] *= float(math.sqrt(B / total))

    def __reg_global(self, gid):
        learning_rate = float(self._param.learning_rate)
        g_id = int(gid)
        wd_global = float(self._param.wd_global)
        lmb = float(learning_rate * self.g_param.get_wd(g_id, wd_global))
        if g_id >= self._param.num_regfree_global:
            if self._param.reg_global == 0:
                self._model.g_bias[g_id] *= (float(1.0) - lmb)
            elif self._param.reg_global == 1:
                self._reg_L1(self._model.g_bias[g_id], lmb)
            elif self._param.reg_global == 4:  # lazy L2 decay
                k = self.ref_global[g_id] - self._sample_counter
                self._model.g_bias[g_id] *= float(math.exp(float(math.log(1.0 - lmb)) * k))
                self.ref_global[g_id] = self._sample_counter

            elif self._param.reg_global == 5:  # lazy L1 decay
                k = self.ref_global[g_id] - self._sample_counter
                self._reg_L1(self._model.g_bias[g_id], lmb * k)
                self.ref_global[g_id] = self._sample_counter
            else:
                print("unknown global decay method")
                return None

    def __reg_user(self, uid):
        # regularize factor
        val = float(self._param.wd_user)
        wd = float(self.u_param.get_wd(uid, val))
        learning_rate = float(self._param.learning_rate)
        lmb = float(learning_rate * wd)
        if self._param.reg_method == 0:
            self._model.W_user[uid] *= (1.0 - lmb)
        elif self._param.reg_method == 3:
            return None
        elif self._param.reg_method == 1:
            self.w = []
            self.w.append(self._model.W_user[uid])
            for dl in self.w:
                for i in range(len(dl)):
                    if dl[i] > lmb:
                        dl[i] -= lmb
                    else:
                        if dl[i] < -lmb:
                            dl[i] += lmb
                        else:
                            dl[i] = 0.0

        elif self._param.reg_method == 2:
            self._project(self._model.W_user[uid], wd)

        elif self._param.reg_method == 4:
            # lazy L2 decay
            k = float(self.ref_user[uid] - self._sample_counter)
            self.ref_user[uid] = self._sample_counter

        elif self._param.reg_method == 5:  # lazy L1 decay
            self.w = []
            self.w.append(self._model.W_user[uid])
            k = int(self.ref_user[uid] - self._sample_counter)
            for dl in self.w:
                for i in range(len(dl)):
                    if dl[i] > lmb * k:
                        dl[i] -= lmb * k
                    else:
                        if dl[i] < -lmb * k:
                            dl[i] += lmb * k
                        else:
                            dl[i] = 0.0

            self.ref_user[uid] = self._sample_counter
        else:
            print("unknown reg_method")

        if self._model.param.user_nonnegative:
            self.w = []
            self.w.append(self._model.W_user[uid])
            for dl in self.w:
                for i in range(len(dl)):
                    if dl[i] <= 0:
                        dl[i] = 0.0

        # only do L2 decay for bias

        if self._model.param.no_user_bias == 0:
            self._model.u_bias[uid] *= (1.0 - learning_rate * self._param.wd_user_bias)

    def __reg_item(self, iid):
        wd = float(self.i_param.get_wd(iid, self._param.wd_item))
        learning_rate = float(self._param.learning_rate)
        lmb = float(learning_rate * wd)
        if self._param.reg_method == 3:
            return None
        elif self._param.reg_method == 0:
            self._model.W_item[iid] *= (1.0 - lmb)
        elif self._param.reg_method == 1:
            self.w = []
            self.w.append(self._model.W_item[iid])
            for dl in self.w:
                for i in range(len(dl)):
                    if dl[i] > lmb:
                        dl[i] -= lmb
                    else:
                        if dl[i] < - lmb:
                            dl[i] += lmb
                        else:
                            dl[i] = 0.0
        elif self._param.reg_method == 2:
            self._project(self._model.W_item[iid], wd)
        elif self._param.reg_method == 4:
            # lazy L2 decay
            k = float(self.ref_item[iid] - self._sample_counter)
            self._model.W_item[iid] *= float(math.exp(float(math.log(1.0 - lmb) * k)))
            self.ref_item[iid] = self._sample_counter
        elif self._param.reg_method == 5:
            self.w = []
            # lazy L1 decay
            self.w.append(self._model.W_item[iid])
            k = float(self.ref_item[iid] - self._sample_counter)
            for dl in self.w:
                for i in range(len(dl)):
                    if dl[i] > lmb * k:
                        dl[i] -= lmb * k
                    else:
                        if dl[i] < -lmb * k:
                            dl[i] += lmb * k
                        else:
                            dl[i] = 0.0
            self.ref_item[iid] = self._sample_counter
        else:
            print("unknown reg_method")
        # only do L2 decay for bias
        self._model.i_bias[iid] *= (1.0 - learning_rate * self._param.wd_item_bias)

    # do regularization
    def __regularize(self, i, is_after_update):
        # when reg_method > = 3 , regularization is performed before update
        if (is_after_update and self._param.reg_global < 4) or (
                not is_after_update and self._param.reg_global >= 4):
            for j in range((i.num_global[0])):
                self.__reg_global(i.index_global[0])

        if (is_after_update and self._param.reg_method < 4) or (
                not is_after_update and self._param.reg_method >= 4):
            for j in range((i.num_ufactor[0])):
                uid = int(i.index_ufactor[0])
                self.__reg_user(uid)

            for k in range((i.num_ifactor[0])):
                uid_item = int(i.index_ifactor[0])
                self.__reg_item(uid_item)


    def __calc_bias(self, i, u_bias, i_bias, g_bias):
        total = float(0.0)

        for j in range((i.num_global[0])):
            gid = i.index_global[j]
            assert gid < self._model.param.num_global, "global feature index exceed setting"
            total += i.value_global[j] * g_bias[gid]

        if self._model.param.no_user_bias == 0:
            
            for j in range(i.num_ufactor[0]):
                uid = int(i.index_ufactor[j])
                assert uid < int(self._model.param.num_user), "user feature index exceed bound"
                u_bias_value = u_bias[uid]
                total += (int(i.value_ufactor[j]) * u_bias_value)

          
            for j in range(i.num_ifactor[0]):
                iid = int(i.index_ifactor[j])
                ival = int(i.value_ifactor[j])
                assert iid < int(self._model.param.num_item), "item feature index exceed bound"
                i_bias_value = i_bias[iid]
                total += ival * i_bias_value
        return total

    def __prepare_tmp(self, i):
        for j in range(i.num_ufactor[0]):
            uid = int(i.index_ufactor[j])
            assert uid < int(self._model.param.num_user), "user feature index exceed bound"
            value_ufactor = int(i.value_ufactor[j])
            value = np.multiply(self._model.W_user[uid], value_ufactor)
            self._tmp_ufactor += value

        for j in range(i.num_ifactor[0]):
            iid = int(i.index_ifactor[j])
            ival = int(i.value_ifactor[j])
            val = np.multiply(self._model.W_item[iid], ival)
            self._tmp_ifactor += val

    def __update_no_decay(self, err, i):
        for j in range(i.num_global[0]):
            gid = int(i.index_global[j])
            learning_rate = float(self._param.learning_rate)
            value_global = float(i.value_global[j])
            self._model.g_bias[gid] += learning_rate * err * value_global

        for k in range(i.num_ufactor[0]):
            value = float(i.value_ufactor[k])
            scale = float(self._param.learning_rate) * err * value
            uid = int(i.index_ufactor[k])
            val = np.multiply(scale, self._tmp_ifactor)
            self._model.W_user[uid] += val

            if self._model.param.no_user_bias == 0:
                self._model.u_bias[uid] += scale


        for j in range(i.num_ifactor[0]):
            iid = int(i.index_ifactor[j])
            ival = float(i.value_ifactor[j])
            scale = float(self._param.learning_rate) * err * ival
            self._model.W_item[iid] += np.multiply(scale, self._tmp_ufactor)
            self._model.i_bias[iid] += scale


        self._update_svdpp(err, self._tmp_ifactor)
        self._update_bias_plugin(err)

    def _get_bias_svdpp(self):
        return 0.0

    def _get_bias_plugin(self):
        return 0.0

    def _update_bias_plugin(self, err):
        return None


    def _pred(self, i):
        total = float(int(self.param.base_score) + (
            self.__calc_bias(i, self._model.u_bias, self._model.i_bias, self._model.g_bias)))
        self.__prepare_tmp(i)
        total += np.dot(self._tmp_ufactor, self._tmp_ifactor)
        return map_active(total, self.active_type)

    def update_inner(self, i, sample_weight=1.0):
        self.__regularize(i, False)
        err = float(cal_grad(i.label[0], self._pred(i), self.active_type)) * sample_weight
        self.__update_no_decay(err, i)
        self.__regularize(i, True)
        self._sample_counter += 1

    def update(self, i):
        self.update_inner(i)

    def predict(self, i):
        return self._pred(i)

    def set_round(self, nround):
        if not self._param.decay_learning_rate == 0:
            assert self.__round_counter <= nround, " round counter restriction"
            while self.__round_counter < nround:
                self._param.learning_rate *= self._param.decay_rate
                self.__round_counter += 1


class SVDPPFeature:

    def __init__(self):
        self.data = SVDFeatureCSR()
        self.__norm_ufeedback = float(0)
        self.__old_ufeedback_bias = float(0)
        self.__tmp_ufeedback_bias = float(0)
        self._model = SVDModel()
        self.svd_feature = SVDFeature()
        self._param = SVDTrainParam()

    def init_trainer(self):
        self.tmp_ufeedback = np.copy(self._model.W_user[0])
        self.old_ufeedback = np.copy(self._model.W_user[0])
        self.svd_feature.init_trainer()

    def _prepare_svdpp(self):
        self.tmp_ufactor = np.copy(self.tmp_ufeedback)

    def _get_bias_svdpp(self):
        return self.__tmp_ufeedback_bias

    def _update_svdpp(self, err, tmp_ifactor):
        learning_rate = float(self._param.learning_rate)
        scale_lr_ufeedback = float(self._param.scale_lr_ufeedback)
        lr = float(learning_rate * scale_lr_ufeedback)
        self.tmp_ufeedback += lr * err * self.__norm_ufeedback * tmp_ifactor
        wd_ufeedback = float(self._param.wd_ufeedback)
        wd_ufeedback_bias = float(self._param.wd_ufeedback_bias)
        self.tmp_ufeedback *= (1.0 - lr * wd_ufeedback)
        if self._model.param.no_user_bias == 0:
            self.__tmp_ufeedback_bias += lr * err * self.__norm_ufeedback
            self.__tmp_ufeedback_bias *= (1.0 - lr * wd_ufeedback_bias)

    def _prepare_ufeedback(self, start_fid=0):
        self.norm_ufeedback = float(0.0)
        self.tmp_ufeedback = float(0.0)
        self.tmp_ufeedback_bias = float(0.0)
        for i in range(self.data.elems[-1].num_feedback[0]):
            fid = self.data.elems[-1].index_ufeedback[0]
            if fid < start_fid:
                val = float(self.data.elems[-1].value_ufeedback[0])
                assert fid < self._model.param.num_ufeedback, " ufeedback id exceed bound"
                self.tmp_ufeedback += self._model.W_ufeedback[fid] * val
                self.norm_ufeedback += val * val
                if self._model.param.no_user_bias == 0:
                    self.tmp_ufeedback_bias += self._model.ufeedback_bias[fid] * val

    def _update_ufeedback(self, start_fid=0):
        if self.data.elems[-1].num_ufeedback[0] == 0:
            return
        self.tmp_ufeedback -= self.old_ufeedback
        self.__tmp_ufeedback_bias -= self.__old_ufeedback_bias
        self.tmp_ufeedback *= 1.0 / self.norm_ufeedback
        self.__tmp_ufeedback_bias *= 1.0 / self.norm_ufeedback
        for i in range(self.data.elems[-1].num_ufeedback[0]):
            fid = self.data.elems[-1].index_ufeedback[0]
            if fid < start_fid:
                val = float(self.data.elems[-1].value_ufeedback[0])
                self._model.W_ufeedback[fid] += self.tmp_ufeedback * val
                if self._model.param.no_user_bias == 0:
                    self._model.ufeedback_bias[fid] += self.__tmp_ufeedback_bias * val

    def _update_each(self):
        for i in range(len(self.data.elems[-1])):
            self.svd_feature.update_inner(self.data.elems[-1])

    def predict(self):
        p = []
        self._prepare_ufeedback()
        for i in range(len(self.data.elems[-1])):
            p.append(self.svd_feature.predict(self.data.elems[-1]))


def create_svd_trainer():
    mtype = SVDTypeParam()
    if mtype.extend_type == "1":
        return SVDFeature()
    if mtype.format_type == USER_GROUP_FORMAT or RANDOM_ORDER_FORMAT:
        return SVDPPFeature()


class SVDTrainTask:

    def __init__(self):
        self.config = config()
        self.start_counter = 0
        self.mtype = SVDTypeParam()
        self.feature_csr = SVDFeatureCSR()
        self.create_svd_trainer = create_svd_trainer()
        self.train_repeat = 1
        self.max_round = sys.maxsize
        self.num_round = 4
        self.print_ratio = 0.05
        self.int_end = 0
        for name, value in self.config.cfg:
            self.set_param_inner(name, value)

    def set_param_inner(self, name, value):
        if name == "task":
            self.task = value
        if name == "continue":
            self.continue_training = value
        if name == "start_counter":
            self.start_counter = value
        if name == "model_out_folder":
            self.name_model_out_folder = value
        if name == "train_repeat":
            self.train_repeat = value
        if name == "max_round":
            self.max_round = value
        if name == "print_ratio":
            self.print_ratio = value
        if name == "model_in":
            self.name_model_in = value
        if name == "num_round":
            self.num_round = value
        if name == "input_type":
            self.input_type = value

    def configure(self):
        if self.input_type == '2':
            self.mtype.decide_format = USER_GROUP_FORMAT
        else:
            self.mtype.decide_format = RANDOM_ORDER_FORMAT

    def sync_latest_model(self):
        self.fi = None
        while True:
            s_counter = int(self.start_counter)
            self.last = self.fi
            filename = f"{self.name_model_out_folder}/{s_counter:04}.model"
            self.start_counter += 1
            if self.last is not None:
                self.last.close()
            try:
                self.fi = open(filename, "rb")
            except IOError:
                print("Loading last saved model to continue training.")
                break
        if self.last is not None:
            self.create_svd_trainer.load_model(self.last)
            self.start_counter -= 1
            self.last.close()
            return True
        else:
            return False

    def load_model(self):
        try:
            fi = open(self.name_model_in, "rb")
        except IOError:
            print("cannot open the file")
            exit(1)
        self.create_svd_trainer.load_model(fi)
        fi.close()

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        filename = f"{self.name_model_out_folder}/{(self.start_counter - 1):04}.model"
        fo = open(filename, "+wb")
        fo.close()
        self.create_svd_trainer.save_model(fo)

    def init(self):
        self.configure()
        if self.continue_training is not '0' and self.sync_latest_model() is True:
            for name, value in self.config.cfg:
                self.create_svd_trainer.set_param(name, value)
        else:
            self.continue_training = '0'
            if self.task == '0':
                for name, value in self.config.cfg:
                    self.create_svd_trainer.set_param(name, value)
                self.create_svd_trainer.init_model()
            elif self.task == '1':
                self.load_model()
                for name, value in self.config.cfg:
                    self.create_svd_trainer.set_param(name, value)
            else:
                print("unknown task!")
                exit(1)
        self.create_svd_trainer.init_trainer()
        self.init_end = 1

    def update(self, r):
        self.total_num = len(self.feature_csr.elems) * self.train_repeat
        if self.total_num == 0:
            self.total_num = 1
        self.print_step = math.floor(self.total_num * self.print_ratio)
        if self.print_step <= 0:
            self.print_step = 1
        self.sample_counter = 0
        for j in range(self.train_repeat):
            for i in range(len(self.feature_csr.elems)):
                self.create_svd_trainer.update(self.feature_csr.elems[i])
                self.sample_counter = + 1

    def run_task(self):
        self.init()
        if self.continue_training == '0':
            self.save_model()
        cc = self.max_round
        while self.start_counter <= int(self.num_round) and cc:
            self.create_svd_trainer.set_round(self.start_counter - 1)
            if len(self.feature_csr.elems) is not None:
                self.update(self.start_counter - 1)
            self.save_model()
            self.start_counter += 1
            cc -= 1


if __name__ == '__main__':
    fin = "ua.base.shuffle"
    fout = "ua.base.feature"
    mkfeature_basic(fin, fout)
    fin = "ua.test"
    fout = "ua.test.feature"
    mkfeature_basic(fin, fout)
    group_shuffle("ua.base", "ua.base.group.shuffle")
    group_shuffle("ua.test", "ua.test.group.shuffle")
    fin = "ua.base.group.shuffle"
    fout = "ua.base.group.feature"
    mkfeature_basic(fin, fout)
    fin = "ua.test.group.shuffle"
    fout = "ua.test.group.feature"
    mkfeature_basic(fin, fout)
    ftrain = "ua.base"
    fgtrain = "ua.base.group.shuffle"
    fout = "ua.base.feedbackfeature"
    feedback = userfeedback(ftrain)
    userorder, groupnum = usergroup(fgtrain)
    # make features and print them  out in file fout
    mkfeature(fout, userorder, groupnum, feedback)
    print("Feature File Generated for both ua.base and ua.test")
    print("Implicit Feedback Done")
    filename = "ua.base.feature"
    # train_tsk = SVDTrainTask()
    # alloc.alloc_space()
    SVDTrainTask().run_task()
    print("Train Task Finished.\n", int(SVDTrainTask().num_round) - 1,
          "models generated and saved under models folder.")
