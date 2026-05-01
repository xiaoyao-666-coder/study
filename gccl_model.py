import torch
import torch.nn as nn

try:
    from convlstm import GConvLSTMCell, ConvLSTMCell
except ImportError:
    from convlstm import GConvLSTMCell

    class ConvLSTMCell(nn.Module):
        def __init__(self, input_dim, hidden_dim, kernel_size, bias):
            super(ConvLSTMCell, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.kernel_size = kernel_size
            self.padding = kernel_size[0] // 2, kernel_size[1] // 2
            self.bias = bias

            self.conv = nn.Conv2d(
                in_channels=self.input_dim + self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            )

        def forward(self, input_tensor, cur_state):
            h_cur, c_cur = cur_state
            combined = torch.cat([input_tensor, h_cur], dim=1)
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
            return (
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            )


class ImageGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImageGraphConv, self).__init__()
        self.weight_matrix = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x_t, adj_matrix):
        batch_size, channels, height, width = x_t.size()
        num_nodes = height * width

        x_flat = x_t.reshape(batch_size, channels, num_nodes)
        x_gcn_list = []

        for channel_idx in range(channels):
            feat_i = x_flat[:, channel_idx, :].reshape(batch_size, num_nodes).transpose(0, 1).contiguous()
            if adj_matrix.is_sparse:
                x_channel = torch.sparse.mm(adj_matrix, feat_i)
            else:
                x_channel = torch.matmul(adj_matrix, feat_i)
            x_gcn_list.append(x_channel.transpose(0, 1).contiguous())

        x_gcn = torch.stack(x_gcn_list, dim=1).reshape(batch_size, channels, height, width)
        return self.weight_matrix(x_gcn)


def build_k_hop_adj_matrix(train_sm_data, theta=7, k=2, valid_mask=None):
    time_steps, height, width = train_sm_data.size()
    num_nodes = height * width
    device = train_sm_data.device

    if valid_mask is None:
        valid_mask = torch.ones((height, width), dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    valid_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        raise ValueError("valid_mask 中没有有效像素，无法构图")

    flat_data = train_sm_data.reshape(time_steps, num_nodes)[:, valid_indices].T
    pearson_corr = torch.corrcoef(flat_data)
    pearson_corr = torch.nan_to_num(pearson_corr, nan=0.0)
    pearson_corr.fill_diagonal_(1.0)

    top_k = min(theta, pearson_corr.size(1))
    _, top_indices = torch.topk(pearson_corr, k=top_k, dim=1)

    valid_adj = torch.zeros((valid_indices.numel(), valid_indices.numel()), device=device)
    row_indices = torch.arange(valid_indices.numel(), device=device).unsqueeze(1).expand_as(top_indices)
    valid_adj[row_indices, top_indices] = 1.0

    valid_adj_tilde = valid_adj + torch.eye(valid_adj.size(0), device=device)
    valid_adj_k = valid_adj_tilde
    for _ in range(k - 1):
        valid_adj_k = torch.matmul(valid_adj_k, valid_adj_tilde)

    valid_adj_k = torch.clamp(valid_adj_k, max=1.0)
    valid_edges = torch.nonzero(valid_adj_k, as_tuple=False)

    full_diag = torch.arange(num_nodes, device=device)
    row_idx = torch.cat([full_diag, valid_indices[valid_edges[:, 0]]])
    col_idx = torch.cat([full_diag, valid_indices[valid_edges[:, 1]]])
    indices = torch.stack([row_idx, col_idx], dim=0)

    adj_matrix = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.size(1), device=device),
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()

    values = torch.ones_like(adj_matrix.values())
    return torch.sparse_coo_tensor(adj_matrix.indices(), values, adj_matrix.size(), device=device).coalesce()


class GCCL_Model(nn.Module):
    def __init__(
        self,
        input_dim,
        gc_dim,
        hidden_dim_g,
        hidden_dim_c,
        kernel_size,
        img_height,
        img_width,
        theta=7,
        k_hop=2,
        output_dim=1,
        bias=True,
    ):
        super(GCCL_Model, self).__init__()
        self.input_dim = input_dim
        self.gc_dim = gc_dim
        self.hidden_dim_g = hidden_dim_g
        self.hidden_dim_c = hidden_dim_c
        self.img_height = img_height
        self.img_width = img_width
        self.theta = theta
        self.k_hop = k_hop
        self.output_dim = output_dim
        self.aux_dim = max(input_dim - output_dim, 0)

        self.graph_conv = ImageGraphConv(input_dim, gc_dim)
        self.gconvlstm_cell = GConvLSTMCell(
            input_dim=input_dim,
            gc_dim=gc_dim,
            hidden_dim_c=hidden_dim_c,
            hidden_dim_g=hidden_dim_g,
            kernel_size=kernel_size,
            bias=bias,
        )
        self.convlstm_cell = ConvLSTMCell(
            input_dim=hidden_dim_g,
            hidden_dim=hidden_dim_c,
            kernel_size=kernel_size,
            bias=bias,
        )
        self.conv_last = nn.Conv2d(hidden_dim_c, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def _to_bcthw(self, frames_tensor):
        if frames_tensor.dim() != 5:
            raise ValueError("输入必须是 5 维张量")

        if frames_tensor.size(-1) == self.input_dim:
            return frames_tensor.permute(0, 1, 4, 2, 3).contiguous()

        if frames_tensor.size(2) == self.input_dim:
            return frames_tensor.contiguous()

        raise ValueError(f"无法识别输入通道维，收到形状 {tuple(frames_tensor.shape)}")

    def _build_decoder_input(self, predicted_smap, aux_context):
        if self.aux_dim == 0:
            return predicted_smap
        return torch.cat([predicted_smap, aux_context], dim=1)

    def forward(self, input_frames, adj_matrix, num_output_frames):
        frames = self._to_bcthw(input_frames)
        batch_size = frames.size(0)

        h_g, c_g = self.gconvlstm_cell.init_hidden(batch_size, (self.img_height, self.img_width))
        h_c, c_c = self.convlstm_cell.init_hidden(batch_size, (self.img_height, self.img_width))

        for t in range(frames.size(1)):
            x_t = frames[:, t]
            gc_t = self.graph_conv(x_t, adj_matrix)
            h_g, c_g = self.gconvlstm_cell(x_t, gc_t, h_c, h_g, c_g)
            h_c, c_c = self.convlstm_cell(h_g, [h_c, c_c])

        aux_context = frames[:, -1, self.output_dim:, :, :] if self.aux_dim > 0 else None
        predictions = []

        x_gen = self.conv_last(h_c)
        predictions.append(x_gen)

        for _ in range(num_output_frames - 1):
            decoder_input = self._build_decoder_input(x_gen, aux_context)
            gc_t = self.graph_conv(decoder_input, adj_matrix)
            h_g, c_g = self.gconvlstm_cell(decoder_input, gc_t, h_c, h_g, c_g)
            h_c, c_c = self.convlstm_cell(h_g, [h_c, c_c])
            x_gen = self.conv_last(h_c)
            predictions.append(x_gen)

        return torch.stack(predictions, dim=1).permute(0, 1, 3, 4, 2).contiguous()

    def predict(self, input_frames, adj_matrix, num_output_frames):
        self.eval()
        with torch.no_grad():
            return self.forward(input_frames, adj_matrix, num_output_frames)
