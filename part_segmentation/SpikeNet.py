import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from pointnet2_ops import pointnet2_utils
from spikingjelly.activation_based.neuron import IFNode,surrogate
def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]	
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True,T=3):
        super(Upsample, self).__init__()
        self.T = T
        self.act1 = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T,1,1,1)
        x = self.act1(x)
        T, B, C, N = x.shape  # x  (T, batch_size, channels, num_points)
        x = x.flatten(0, 1)  # (T * batch_size, channels, num_points)
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.reshape(T, B, -1, N).contiguous()
        x = x.mean(dim=0)
        return x

class DSSR(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True,T=3):
        super(DSSR, self).__init__()
        self.T = T
        self.act1 = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)
        self.conv1 = nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion), kernel_size=kernel_size, bias=bias)
        self.bn1 = nn.BatchNorm1d(int(channel * res_expansion))

        self.act2 = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)
        self.conv2 = nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion), kernel_size=kernel_size, bias=bias)
        self.bn2 = nn.BatchNorm1d(int(channel * res_expansion))
        

    def forward(self, x):
        xx = x
        x = x.unsqueeze(0).repeat(self.T,1,1,1)
        x = self.act1(x)
        T, B, C, N = x.shape  # x  (T, batch_size, channels, num_points)
        x = x.flatten(0, 1)  #  (T * batch_size, channels, num_points)

        x = self.conv1(x)
        x = self.bn1(x)
        x = x.reshape(T, B, -1, N).contiguous()
        x = self.act2(x)
        x = x.flatten(0, 1)  #  (T * batch_size, channels, num_points)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x.reshape(T, B, -1, N).contiguous()
        x = x.mean(dim=0)
        return x  + xx
    
class DSR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True,T=3):
        super(DSR, self).__init__()
        self.T = T
        self.act1 = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_channels)
       

    def forward(self, x):
        xx = x
        x = x.unsqueeze(0).repeat(self.T,1,1,1)
        x = self.act1(x)
        T, B, C, N = x.shape  # x  (T, batch_size, channels, num_points)
        x = x.flatten(0, 1)  #  (T * batch_size, channels, num_points)
        x = self.conv1(x)
        x = self.bn1(x) 
        x = x.reshape(T, B, -1, N).contiguous()
        x = x.mean(dim=0)
        return x  + xx

class SVMT(nn.Module):
    def __init__(self, channel,kernel_size=1, groups=1, res_expansion=1.0, bias=True,T=3):
        super(SVMT, self).__init__()
        self.T = T
        self.act11 = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)
        self.q_conv = nn.Conv1d(channel, int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias)
        self.q_bn = nn.BatchNorm1d(int(channel * res_expansion) )
        self.q_if = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)


        self.k_conv = nn.Conv1d(channel,int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias)
        self.k_bn = nn.BatchNorm1d(int(channel * res_expansion))
        self.k_if = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)
 

        self.v_conv = nn.Conv1d(channel, int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias)
        self.v_bn = nn.BatchNorm1d(int(channel * res_expansion))
        self.v_if = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)

        self.actif1 = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)
        self.trans_conv = nn.Conv1d(channel, int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias)
        self.after_norm = nn.BatchNorm1d(int(channel * res_expansion))
        self.actif2 = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)
        self.actif3 = IFNode(v_threshold=1.0,surrogate_function=surrogate.ATan(),step_mode='m',detach_reset=True)


    def forward(self, x):
        xx = x
        x = x.unsqueeze(0).repeat(self.T,1,1,1)
        x = self.act11(x)
        T, B, C, N = x.shape  # x  (T, batch_size, channels, num_points)
        xxx = x
        x = x.flatten(0, 1)  #  (T * batch_size, channels, num_points)

        x_q = self.q_if(self.q_bn(self.q_conv(x)).reshape(T, B, -1, N).contiguous()).reshape(T * B, -1, N)
        

        x_k = self.k_if(self.k_bn(self.k_conv(x)).reshape(T, B, -1, N).contiguous()).reshape(T * B, -1, N)
        

        x_v = self.v_if(self.v_bn(self.v_conv(x)).reshape(T, B, -1, N).contiguous())#.reshape(T * B, -1, N) 
        

        q1 = x_q - x_k
        q1 = q1.reshape(T, B, -1, N).contiguous()
        attn = self.actif1(q1)

        v = torch.sum(x_v,dim=2,keepdim=True)
        
        x_v = self.actif2(v)
        x_r = torch.mul(attn,x_v)

        x_r = self.actif3(xxx-x_r).flatten(0, 1)
       

        x_r= self.after_norm(self.trans_conv(x_r)).reshape(T, B, -1, N).contiguous()
        x_r = x_r.mean(dim=0)
        x = x_r + xx

        return x

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        # self.act1 = IFNode(v_threshold=0.5,surrogate_function=surrogate.ATan(),step_mode='s',detach_reset=True)
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 use_xyz=True,T=3):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer1 = DSR(in_channels, out_channels, bias=bias,T=T)
        operation = []
        for _ in range(blocks):
            operation.append(
                DSSR(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias,T=T)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        # print(f'x1:{x.shape}')
        x = x.permute(0, 1, 3, 2)
        # print(f'x2:{x.shape}')
        x = x.reshape(-1, d, s)
        # print(f'x3:{x.shape}')
        x = self.transfer1(x)
        
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True,T=3):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                SVMT(channels, groups=groups, res_expansion=res_expansion, bias=bias,T=T)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = Upsample(in_channel, out_channel, 1, bias=bias)
        self.extraction = DSSR(out_channel, groups=groups, res_expansion=res_expansion,
                                bias=bias)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points




class spikeNet(nn.Module):
    def __init__(self, num_classes=50,points=2048, embed_dim=32, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[1, 1, 1, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[2,2,2,2],
                 gmp_dim=64,cls_dim=64, **kwargs):
        super(spikeNet, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = num_classes
        self.points = points
        self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]
        ### Building Encoder #####
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias,use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)


        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias)
            )

        self.act = get_activation(activation)

        # class label mapping
        self.cls_map = nn.Sequential(
            Upsample(16, cls_dim, bias=bias),
            Upsample(cls_dim, cls_dim, bias=bias)
        )
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(Upsample(en_dim, gmp_dim, bias=bias))
        self.gmp_map_end = Upsample(gmp_dim*len(en_dims), gmp_dim, bias=bias)

        # classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias)
        )
        self.en_dims = en_dims

    def forward(self, x, norm_plt, cls_label):
        xyz = x.permute(0, 2, 1)
        x = torch.cat([x,norm_plt],dim=1)
        x = self.embedding(x)  # B,D,N

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            xyz_list.append(xyz)
            x_list.append(x)

        # here is the decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)

        # here is the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # [b, gmp_dim, 1]

        #here is the cls_token
        cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))  # [b, cls_dim, 1]
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


def SpikeNet(num_classes=50, **kwargs) -> spikeNet:
    return spikeNet(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2,2,2,2], pre_blocks=[1,1,1,1], pos_blocks=[1,1,1,1],
                 k_neighbors=[32,32,32,32], reducers=[4,4,4,4],
                 de_dims=[512,256,128,128], de_blocks=[4,4,4,4],
                 gmp_dim=64,cls_dim=64, **kwargs)


