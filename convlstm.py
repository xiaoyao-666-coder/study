
import torch.nn as nn
import torch


class GConvLSTMCell(nn.Module):
    """
    Graph ConvLSTM Cell based on paper formulas (7) ~ (12).
    
    Compared to standard ConvLSTM, GConvLSTM concatenates 4 features as input:
    - X^t: current input
    - GC_t^{K}: graph convolution extracted spatial features
    - H_c^{t-1}: previous hidden state from upper layer (ConvLSTM)
    - H_g^{t-1}: previous hidden state from current layer (GConvLSTM)
    """

    def __init__(self, input_dim, gc_dim, hidden_dim_c, hidden_dim_g, kernel_size, bias):
        """
        Initialize GConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor X^t.
        gc_dim: int
            Number of channels of graph convolution features GC_t^{K}.
        hidden_dim_c: int
            Number of channels of hidden state from upper ConvLSTM layer H_c.
        hidden_dim_g: int
            Number of channels of hidden state from current GConvLSTM layer H_g.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(GConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.gc_dim = gc_dim
        self.hidden_dim_c = hidden_dim_c
        self.hidden_dim_g = hidden_dim_g

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Total input channels: X + GC + H_c + H_g
        total_input_dim = input_dim + gc_dim + hidden_dim_c + hidden_dim_g
        
        # Main convolution for input, forget, output gates and candidate memory
        self.conv = nn.Conv2d(in_channels=total_input_dim,
                              out_channels=4 * hidden_dim_g,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        # Peephole connections: W_ci, W_cf, W_co for cell state
        # These are element-wise multiplications ( Hadamard product )
        self.W_ci = nn.Parameter(torch.zeros(1, hidden_dim_g, 1, 1))
        self.W_cf = nn.Parameter(torch.zeros(1, hidden_dim_g, 1, 1))
        self.W_co = nn.Parameter(torch.zeros(1, hidden_dim_g, 1, 1))

    def forward(self, x_t, gc_t, h_c_prev, h_g_prev, c_g_prev):
        """
        Forward pass of GConvLSTM cell following formula (7) ~ (12).

        Parameters
        ----------
        x_t: torch.Tensor
            Current input tensor X^t, shape (B, input_dim, H, W)
        gc_t: torch.Tensor
            Graph convolution features GC_t^{K}, shape (B, gc_dim, H, W)
        h_c_prev: torch.Tensor
            Previous hidden state from upper ConvLSTM layer H_c^{t-1}, shape (B, hidden_dim_c, H, W)
        h_g_prev: torch.Tensor
            Previous hidden state from current GConvLSTM layer H_g^{t-1}, shape (B, hidden_dim_g, H, W)
        c_g_prev: torch.Tensor
            Previous cell state from current GConvLSTM layer C_g^{t-1}, shape (B, hidden_dim_g, H, W)

        Returns
        -------
        h_g_next: torch.Tensor
            Next hidden state H_g^t, shape (B, hidden_dim_g, H, W)
        c_g_next: torch.Tensor
            Next cell state C_g^t, shape (B, hidden_dim_g, H, W)
        """
        # Concatenate features along channel axis: [X^t, GC_t^{K}, H_c^{t-1}, H_g^{t-1}]
        combined = torch.cat([x_t, gc_t, h_c_prev, h_g_prev], dim=1)

        # Convolution
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim_g, dim=1)

        # Apply gates with peephole connections following formula (7) ~ (10)
        # Input gate: I_t = sigmoid(cc_i + W_ci * C_g^{t-1})
        i = torch.sigmoid(cc_i + self.W_ci * c_g_prev)
        
        # Forget gate: F_t = sigmoid(cc_f + W_cf * C_g^{t-1})
        f = torch.sigmoid(cc_f + self.W_cf * c_g_prev)
        
        # Candidate memory: G_t = tanh(cc_g)
        g = torch.tanh(cc_g)
        
        # Cell state update: C_g^t = F_t * C_g^{t-1} + I_t * G_t
        c_g_next = f * c_g_prev + i * g
        
        # Output gate: O_t = sigmoid(cc_o + W_co * C_g^t)
        o = torch.sigmoid(cc_o + self.W_co * c_g_next)
        
        # Hidden state: H_g^t = O_t * tanh(C_g^t)
        h_g_next = o * torch.tanh(c_g_next)

        return h_g_next, c_g_next

    def init_hidden(self, batch_size, image_size):
        """Initialize hidden and cell states."""
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim_g, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim_g, height, width, device=self.conv.weight.device))


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
