from __future__ import division
from __future__ import print_function


from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

from attention import attention_luong
from attention import attention_nmts_fast


class NMTSDecoderCellOld(rnn_cell.RNNCell):
    def __init__(self, cells):
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        self._cells = cells

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        # TODO: need nice way to incorporate attention state
        state = []
        for cell in self._cells:
            state.append(cell.zero_state(batch_size, dtype))
        return tuple(state)

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "nmts_decoder_cell"):
            states, c_t = state
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with vs.variable_scope("cell_{}".format(i)):
                    cur_state = states[i]
                    prev_inp = cur_inp

                    h_dim = cur_inp.get_shape().with_rank(2)[1].value
                    Wp = vs.get_variable("Wp", [2*h_dim, h_dim])
                    bp = vs.get_variable("bp", [h_dim])
                    cur_inp = math_ops.matmul(array_ops.concat(1, [cur_inp, c_t]), Wp) + bp
                    cur_state = rnn_cell.LSTMStateTuple(cur_state.c, cur_inp)

                    next_inp, new_state = cell(cur_inp, cur_state)
                    cur_inp = prev_inp + next_inp if i < len(self._cells) - 1 else next_inp
                    new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, (new_states, c_t)


class NMTSDecoderCell(rnn_cell.RNNCell):
    def __init__(self, cells, attention="luong"):
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        self._cells = cells
        self._attention = attention

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        # TODO: need nice way to incorporate attention state
        state = []
        for cell in self._cells:
            state.append(cell.zero_state(batch_size, dtype))
        return tuple(state)

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "nmts_decoder_cell"):
            states, encoder_hs = state
            cur_inp = inputs
            new_states = []

            with vs.variable_scope("cell_0"):
                cur_inp, cur_state = self._cells[0](cur_inp, states[0])
                if self._attention == "luong":
                    c_t = attention_luong(cur_inp, encoder_hs)
                elif self._attention == "nmts":
                    c_t = attention_nmts_fast(cur_inp, encoder_hs)
                else:
                    raise ValueError("Unknown attention type: {}".format(self._attention))

            new_states.append(cur_state)
            states = states[1:]
            for i, cell in enumerate(self._cells[1:]):
                with vs.variable_scope("cell_{}".format(i+1)):
                    cur_state = states[i]
                    prev_inp = cur_inp

                    h_dim = cur_inp.get_shape().with_rank(2)[1].value
                    Wp = vs.get_variable("Wp", [2*h_dim, h_dim])
                    bp = vs.get_variable("bp", [h_dim])
                    cur_inp = math_ops.matmul(array_ops.concat(1, [cur_inp, c_t]), Wp) + bp
                    cur_state = rnn_cell.LSTMStateTuple(cur_state.c, cur_inp)

                    next_inp, new_state = cell(cur_inp, cur_state)
                    cur_inp = prev_inp + next_inp if i < len(self._cells[1:]) - 1 else next_inp
                    new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, (new_states, encoder_hs)


class MultiSkipRNNCell(rnn_cell.RNNCell):
    def __init__(self, cells):
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        self._cells = cells

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "multi_skip_cell"):
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with vs.variable_scope("cell_{}".format(i)):
                    cur_state = state[i]
                    prev_inp = cur_inp
                    next_inp, new_state = cell(cur_inp, cur_state)
                    cur_inp = prev_inp + next_inp if i < len(self._cells) - 1 else next_inp
                    new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, new_states
