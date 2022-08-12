import os
import sys
import numpy as np
import unittest

import deepmd.op
from deepmd.env import tf
from deepmd.env import op_module
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION


def qr(x, nbit):
    return np.round(x * 2**nbit) / (2**nbit)


def qf(x, nbit):
    return np.floor(x * 2**nbit) / (2**nbit)


class TestNvnmdMapOp(tf.test.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = "dpparallel"
        self.sess = self.test_session(config=config).__enter__()
        self.nbit_x = 14
        self.nbit_xk = 10
        self.nbit_yk = 10
        self.prec_x = 1.0/2**self.nbit_x
        self.prec_xk = 1.0/2**self.nbit_xk
        self.prec_yk = 1.0/2**self.nbit_yk

    def gen_map_table(self):
        n = 2**self.nbit_xk
        #
        t = np.arange(n+1) / n
        v = np.power(t, 3)
        dv = 3 * np.power(t, 2)
        dv2 = 6 * np.power(t, 1)
        self.v = qr(v, self.nbit_yk).reshape([-1, 1])
        self.dv = qr(dv, self.nbit_yk).reshape([-1, 1])
        self.dv2 = qr(dv2, self.nbit_yk).reshape([-1, 1])

    def map_nvnmd_py(self, x, v, d_v, dv, d_dv, prec_xk, nbit_yk):
        pv = []
        for ii in range(len(x)):
            # y = vk + dvk * dxk
            k = np.int32(np.floor(x[ii] / prec_xk))
            xk = k * prec_xk
            dxk = x[ii] - xk
            vk = v[k]
            dvk = d_v[k]
            pvi = vk + dvk*dxk
            pv.append(pvi)
        pv = np.array(pv).reshape([-1, 1])
        return pv

    def test_map_op(self):
        self.gen_map_table()
        x = qr(np.random.rand(100)*0.9, self.nbit_x).reshape([-1, 1])
        y = self.map_nvnmd_py(x, self.v, self.dv, self.dv, self.dv2, self.prec_xk, self.nbit_yk)
        ty = op_module.map_nvnmd(x, self.v, self.dv, self.dv, self.dv2, self.prec_xk, self.nbit_yk)
        self.sess.run(tf.global_variables_initializer())
        typ = self.sess.run(ty)
        np.testing.assert_almost_equal(typ, y, 5)


class TestNvnmdMatmulOp(tf.test.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = "dpparallel"
        self.sess = self.test_session(config=config).__enter__()
        self.nbit = 13

    def nvnmd_matmul_py(self, x, w, is_round, nbit, nbit2, nbit3):
        if nbit < 0:
            return np.matmul(x, w)
        else:
            if is_round == 1:
                return qr(np.matmul(x, w), nbit)
            else:
                return qf(np.matmul(x, w), nbit)

    def test_nvnmd_matmul(self):
        N = 10
        M = 10
        K = 10
        x = np.random.rand(N, M)
        w = np.random.rand(M, K)
        x = qr(x, self.nbit)
        w = qr(x, self.nbit)
        y = self.nvnmd_matmul_py(x, w, 1, self.nbit, self.nbit, -1)
        ty = op_module.matmul_nvnmd(x, w, 1, self.nbit, self.nbit, -1)
        self.sess.run(tf.global_variables_initializer())
        typ = self.sess.run(ty)
        np.testing.assert_almost_equal(typ, y, 5)


class TestNvnmdQuantizeOp(tf.test.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = "dpparallel"
        self.sess = self.test_session(config=config).__enter__()
        self.nbit = 13

    def nvnmd_quantize_py(self, x, is_round, nbit, nbit2, nbit3):
        if nbit < 0:
            return x
        else:
            if is_round == 1:
                return qr(x, nbit)
            else:
                return qf(x, nbit)

    def test_nvnmd_quantize(self):
        N = 10
        M = 10
        x = np.random.rand(N, M)
        y = self.nvnmd_quantize_py(x, 1, self.nbit, self.nbit, -1)
        ty = op_module.quantize_nvnmd(x, 1, self.nbit, self.nbit, -1)
        self.sess.run(tf.global_variables_initializer())
        typ = self.sess.run(ty)
        np.testing.assert_almost_equal(typ, y, 5)


class TestNvnmdTanh2Op(tf.test.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = "dpparallel"
        self.sess = self.test_session(config=config).__enter__()
        self.nbit = 13

    def nvnmd_tanh2_py(self, x, is_round, nbit, nbit2, nbit3):
        if nbit < 0:
            x1 = np.clip(x, -2, 2)
            x2 = np.clip(x, -4, 4)
            x1a = np.abs(x1)
            x2a = np.abs(x2)
            y1 = x1a - x1a * x1a * 0.25
            y2 = x2a * 0.03125 - x2a * x2a * 0.00390625
            y = y1 + y2
            y = y * np.sign(x1)
        else:
            if is_round:
                x = qr(x, nbit)
                x1 = np.clip(x, -2, 2)
                x2 = np.clip(x, -4, 4)
                x1a = np.abs(x1)
                x2a = np.abs(x2)
                y1 = x1a - x1a * x1a * 0.25
                y2 = x2a * 0.03125 - x2a * x2a * 0.00390625
                y = qr(y1, nbit) + qr(y2, nbit)
                y = y * np.sign(x1)
            else:
                x = qf(x, nbit)
                x1 = np.clip(x, -2, 2)
                x2 = np.clip(x, -4, 4)
                x1a = np.abs(x1)
                x2a = np.abs(x2)
                y1 = x1a - x1a * x1a * 0.25
                y2 = x2a * 0.03125 - x2a * x2a * 0.00390625
                y = qf(y1, nbit) + qf(y2, nbit)
                y = y * np.sign(x1)
        return y

    def test_nvnmd_tanh2(self):
        N = 10
        M = 10
        x = np.random.rand(N, M)
        y = self.nvnmd_tanh2_py(x, 1, self.nbit, self.nbit, -1)
        ty = op_module.tanh2_nvnmd(x, 1, self.nbit, self.nbit, -1)
        self.sess.run(tf.global_variables_initializer())
        typ = self.sess.run(ty)
        np.testing.assert_almost_equal(typ, y, 5)


class TestNvnmdTanh4Op(tf.test.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = "dpparallel"
        self.sess = self.test_session(config=config).__enter__()
        self.nbit = 13

    def nvnmd_tanh4_py(self, x, is_round, nbit, nbit2, nbit3):
        if nbit < 0:
            x = np.clip(x, -2, 2)
            xa = np.abs(x)
            xx = x * x
            y = xx * (xx * 0.0625 - xa * 0.25) + xa
            y = y * np.sign(x)
            return y
        else:
            if is_round:
                x = np.clip(x, -2, 2)
                xa = np.abs(x)
                xx = qr(x * x, nbit)
                y = xx * (xx * 0.0625 - xa * 0.25) + xa
                y = qr(y, nbit)
                y = y * np.sign(x)
                return y
            else:
                x = np.clip(x, -2, 2)
                xa = np.abs(x)
                xx = qf(x * x, nbit)
                y = xx * (xx * 0.0625 - xa * 0.25) + xa
                y = qf(y, nbit)
                y = y * np.sign(x)
                return y

    def test_nvnmd_tanh4(self):
        N = 10
        M = 10
        x = np.random.rand(N, M)
        y = self.nvnmd_tanh4_py(x, 1, self.nbit, self.nbit, -1)
        ty = op_module.tanh4_nvnmd(x, 1, self.nbit, self.nbit, -1)
        self.sess.run(tf.global_variables_initializer())
        typ = self.sess.run(ty)
        np.testing.assert_almost_equal(typ, y, 5)


class TestProdEnvMatANvnmdQuantize(tf.test.TestCase):
    def setUp(self):
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = "dpparallel"
        self.sess = self.test_session(config=config).__enter__()

    def prod_env_mat_a_nvnmd_quantize_py(self):
        coord = [
            12.83, 2.56, 2.18,
            12.09, 2.87, 2.74,
            0.25, 3.32, 1.68,
            3.36, 3.00, 1.81,
            3.51, 2.51, 2.60,
            4.27, 3.22, 1.56]
        coord = np.reshape(np.array(coord), [1, -1])
        #
        atype = [
            0, 1, 1,
            0, 1, 1]
        atype = np.reshape(np.array(atype), [1, -1])
        #
        box = [
            13., 0., 0.,
            0., 13., 0.,
            0., 0., 13.]
        box = np.reshape(np.array(box), [1, -1])
        #
        natoms = [6, 6, 3, 3]
        natoms = np.int32(np.array(natoms))
        #
        mesh = np.int32(np.array([0, 0, 0, 2, 2, 2]))
        t_avg = np.zeros([2, 6*4])
        t_std = np.ones([2, 6*4])
        #
        y = [
            1.279150390625000000e+01, 3.530029296875000000e+00, 4.399414062500000000e-01, -3.699951171875000000e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 9.572753906250000000e-01, -7.399902343750000000e-01, 3.100585937500000000e-01, 5.600585937500000000e-01, 1.004028320312500000e+00, 4.200439453125000000e-01, 7.600097656250000000e-01, -5.000000000000000000e-01, 1.372167968750000000e+01, 3.680053710937500000e+00, -5.004882812500000000e-02, 4.200439453125000000e-01, 2.053308105468750000e+01, 4.439941406250000000e+00, 6.600341796875000000e-01, -6.199951171875000000e-01,
            9.572753906250000000e-01, 7.399902343750000000e-01, -3.100585937500000000e-01, -5.600585937500000000e-01, 1.911486816406250000e+01, 4.270019531250000000e+00, 1.300048828125000000e-01, -9.300537109375000000e-01, 2.671752929687500000e+00, 1.160034179687500000e+00, 4.499511718750000000e-01, -1.060058593750000000e+00, 1.968591308593750000e+01, 4.420043945312500000e+00, -3.599853515625000000e-01, -1.400146484375000000e-01, 2.834790039062500000e+01, 5.180053710937500000e+00, 3.499755859375000000e-01, -1.180053710937500000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
            1.004028320312500000e+00, -4.200439453125000000e-01, -7.600097656250000000e-01, 5.000000000000000000e-01, 9.791259765625000000e+00, 3.109985351562500000e+00, -3.199462890625000000e-01, 1.300048828125000000e-01, 2.671752929687500000e+00, -1.160034179687500000e+00, -4.499511718750000000e-01, 1.060058593750000000e+00, 1.213024902343750000e+01, 3.260009765625000000e+00, -8.100585937500000000e-01, 9.200439453125000000e-01, 1.618493652343750000e+01, 4.020019531250000000e+00, -9.997558593750000000e-02, -1.199951171875000000e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
            1.279150390625000000e+01, -3.530029296875000000e+00, -4.399414062500000000e-01, 3.699951171875000000e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 8.867187500000000000e-01, 1.500244140625000000e-01, -4.899902343750000000e-01, 7.900390625000000000e-01, 9.389648437500000000e-01, 9.100341796875000000e-01, 2.199707031250000000e-01, -2.500000000000000000e-01, 9.791259765625000000e+00, -3.109985351562500000e+00, 3.199462890625000000e-01, -1.300048828125000000e-01, 1.911486816406250000e+01, -4.270019531250000000e+00, -1.300048828125000000e-01, 9.300537109375000000e-01,
            8.867187500000000000e-01, -1.500244140625000000e-01, 4.899902343750000000e-01, -7.900390625000000000e-01, 1.372167968750000000e+01, -3.680053710937500000e+00, 5.004882812500000000e-02, -4.200439453125000000e-01, 2.163330078125000000e+00, 7.600097656250000000e-01, 7.099609375000000000e-01, -1.040039062500000000e+00, 1.213024902343750000e+01, -3.260009765625000000e+00, 8.100585937500000000e-01, -9.200439453125000000e-01, 1.968591308593750000e+01, -4.420043945312500000e+00, 3.599853515625000000e-01, 1.400146484375000000e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
            9.389648437500000000e-01, -9.100341796875000000e-01, -2.199707031250000000e-01, 2.500000000000000000e-01, 2.053308105468750000e+01, -4.439941406250000000e+00, -6.600341796875000000e-01, 6.199951171875000000e-01, 2.163330078125000000e+00, -7.600097656250000000e-01, -7.099609375000000000e-01, 1.040039062500000000e+00, 1.618493652343750000e+01, -4.020019531250000000e+00, 9.997558593750000000e-02, 1.199951171875000000e-01, 2.834790039062500000e+01, -5.180053710937500000e+00, -3.499755859375000000e-01, 1.180053710937500000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00]
        y = np.array(y).reshape([1, -1])
        # 
        dy = [
            -7.060058593750000000e+00, -8.798828125000000000e-01, 7.399902343750000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.479980468750000000e+00, -6.201171875000000000e-01, -1.120117187500000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -8.400878906250000000e-01, -1.520019531250000000e+00, 1.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -7.360107421875000000e+00, 1.000976562500000000e-01, -8.400878906250000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -8.879882812500000000e+00, -1.320068359375000000e+00, 1.239990234375000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00,
            -1.479980468750000000e+00, 6.201171875000000000e-01, 1.120117187500000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -8.540039062500000000e+00, -2.600097656250000000e-01, 1.860107421875000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -2.320068359375000000e+00, -8.999023437500000000e-01, 2.120117187500000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -8.840087890625000000e+00, 7.199707031250000000e-01, 2.800292968750000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -1.036010742187500000e+01, -6.999511718750000000e-01, 2.360107421875000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
            8.400878906250000000e-01, 1.520019531250000000e+00, -1.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -6.219970703125000000e+00, 6.398925781250000000e-01, -2.600097656250000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 2.320068359375000000e+00, 8.999023437500000000e-01, -2.120117187500000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -6.520019531250000000e+00, 1.620117187500000000e+00, -1.840087890625000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -8.040039062500000000e+00, 1.999511718750000000e-01, 2.399902343750000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
            7.060058593750000000e+00, 8.798828125000000000e-01, -7.399902343750000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -3.000488281250000000e-01, 9.799804687500000000e-01, -1.580078125000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -1.820068359375000000e+00, -4.399414062500000000e-01, 5.000000000000000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 6.219970703125000000e+00, -6.398925781250000000e-01, 2.600097656250000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 8.540039062500000000e+00, 2.600097656250000000e-01, -1.860107421875000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00,
            3.000488281250000000e-01, -9.799804687500000000e-01, 1.580078125000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 7.360107421875000000e+00, -1.000976562500000000e-01, 8.400878906250000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, -1.520019531250000000e+00, -1.419921875000000000e+00, 2.080078125000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 6.520019531250000000e+00, -1.620117187500000000e+00, 1.840087890625000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 8.840087890625000000e+00, -7.199707031250000000e-01, -2.800292968750000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
            1.820068359375000000e+00, 4.399414062500000000e-01, -5.000000000000000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 8.879882812500000000e+00, 1.320068359375000000e+00, -1.239990234375000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 1.520019531250000000e+00, 1.419921875000000000e+00, -2.080078125000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 8.040039062500000000e+00, -1.999511718750000000e-01, -2.399902343750000000e-01, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 1.036010742187500000e+01, 6.999511718750000000e-01, -2.360107421875000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00]
        dy = np.array(dy).reshape([1, -1])
        return coord, atype, natoms, box, mesh, t_avg, t_std, y, dy

    def test_prod_env_mat_a_nvnmd_quantize(self):
        coord, atype, natoms, box, mesh, t_avg, t_std, y, dy = self.prod_env_mat_a_nvnmd_quantize_py()
        ty, tdy, trij, tnlist = op_module.prod_env_mat_a_nvnmd_quantize(
            coord,
            atype,
            natoms,
            box,
            mesh,
            t_avg,
            t_std,
            rcut_a=0,
            rcut_r=6.0,
            rcut_r_smth=0.5,
            sel_a=[2, 4],
            sel_r=[0, 0]
        )
        self.sess.run(tf.global_variables_initializer())
        typ, tdyp, trijp, tnlistp = self.sess.run([ty, tdy, trij, tnlist])
        np.testing.assert_almost_equal(typ, y, 5)
        np.testing.assert_almost_equal(tdyp, dy, 5)


if __name__ == '__main__':
    unittest.main()