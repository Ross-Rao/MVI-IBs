# python import
# package import
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans  # pip install fast-pytorch-kmeans
# local import
from custom.models.unet import UNet64
from custom.models.loss import PerceptualLoss
from custom.models.adaptive_clustering_transformer import AcTransformerEncoder


__all__ = ['Stage1Model', 'Stage1ModelLoss', 'Stage2Model', 'Stage2ModelLoss']


class ViTSequentialClassificationHead(nn.Sequential):
    def __init__(self, d_model, n_classes):
        super().__init__(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)


class MemoryBank(torch.autograd.Function):
    """
    update the memory bank with the new data
    """

    @staticmethod
    def inf(x, memory, params):
        # pytorch lightning use autograd
        # if you don't want to calculate gradient, you can use inf.
        t = params[0]
        output = x @ memory.t() / t
        return output

    @staticmethod
    def forward(ctx, x, index, memory, params):
        # only train sample's index should be stored to update memory
        # if index is None, it should be external test samples
        # if index is not None and index.max() < memory.size(0), it should be val samples
        if index is not None and index.max() < memory.size(0):
            # use clone to avoid modifying input(x and index) in multiprocess environment
            ctx.save_for_backward(x.clone(), index.clone(), memory, params)
        output = MemoryBank.inf(x, memory, params)
        return output  # only x is needed for backward

    @staticmethod
    def backward(ctx, *grad_output):
        x, index, memory, params = ctx.saved_tensors
        t, momentum = params[0], params[1]

        grad_x = grad_output[0]
        grad_input = grad_x @ memory / t

        # update memory
        weight = memory[index] * momentum + x * (1. - momentum)
        memory[index] = F.normalize(weight, p=2, dim=1)  # normalize memory

        return grad_input, None, None, None


class MemoryCluster(nn.Module):
    """
    use memory cluster to calculate the instance loss and anchor loss
    """

    def __init__(self, n_samples, npc_dim, neighbors=3, n_cluster=10, temperature=0.07, momentum=0.5, const=0.):
        super().__init__()
        self.samples_num = n_samples
        self.npc_dim = npc_dim
        self.neighbors_num = neighbors
        self.const = const
        self.n_cluster = n_cluster

        # the memory bank should only store train samples
        # indices are reset by order: train, val, test
        # so if input index > n_samples, it should be val or internal test samples
        # if input index is None, it should be external test samples
        # it works by:
        # 1. `split_dataset_folds` by param `reset_split_index` in modules/monai_data_module.py
        # 2. `IndexTransformd` in utils/custom_transforms.py
        self.register_buffer('memory', torch.rand(n_samples, npc_dim))
        std = 1. / torch.sqrt(torch.tensor(npc_dim / 3.))
        self.memory = self.memory * 2 * std - std
        self.register_buffer('params', torch.tensor([temperature, momentum]))

        # anchor samples' flag >= 0, instance samples' flag < 0
        self.register_buffer('flag', -torch.arange(n_samples).long() - 1)

        # neighbors
        self.register_buffer('neighbors', torch.LongTensor(n_samples, neighbors))

        self.register_buffer('entropy', torch.zeros(n_samples))  # use local variable might error for device

        # cluster
        self.kmeans = KMeans(n_clusters=self.n_cluster, mode='cosine')
        # self.spectral = SpectralClustering(n_clusters=self.n_cluster, affinity='nearest_neighbors',
        #                                    n_neighbors=neighbors * 2,
        #                                    assign_labels='discretize', random_state=42)
        self.register_buffer('centroids', torch.zeros(n_cluster, npc_dim))
        self.autograd = nn.Parameter(torch.tensor(1.0))  # used to control autograd

    def update_anchor(self, search_rate, mini_batch_size=256):
        with torch.no_grad():
            # calculate entropy, use for loop to save gpu memory
            for start in range(0, self.samples_num, mini_batch_size):
                end = min(start + mini_batch_size, self.samples_num)
                # sim = MemoryBank.apply(self.memory[start:end], None, self.memory, self.params)
                sim = MemoryBank.inf(self.memory[start:end], self.memory, self.params)
                pred = F.softmax(sim, dim=1)  # mini_batch_size * n_samples
                self.entropy[start:end] = -torch.sum(pred * torch.log(pred + self.const), dim=1)

            # update anchor
            anchor_nums = int(self.samples_num * search_rate)  # search range expand with epoch
            sort = torch.argsort(self.entropy)
            anchor = sort[:anchor_nums]  # Lower entropy indicates a higher similarity
            instance = sort[anchor_nums:]
            self.flag[anchor] = 1
            self.flag[instance] = -1

            # using anchor feature to find neighbors, for loop to save gpu memory
            for start in range(0, anchor_nums, mini_batch_size):
                end = min(start + mini_batch_size, anchor_nums)
                anchor_feature = self.memory[anchor[start:end]]
                sim = MemoryBank.inf(anchor_feature, self.memory, self.params)
                pred = F.softmax(sim, dim=1)
                # remove self-similarity
                pred[torch.arange(end - start), anchor[start:end]] = -1
                # find the nearest neighbor (should be written by torch.max)
                _, top_k_indices = torch.topk(pred, k=self.neighbors_num, dim=1)
                self.neighbors[anchor[start:end]] = top_k_indices

    def forward(self, zp, index, local_neighbor_indices=None):
        if index is not None and index.max() < self.samples_num:  # train sample to anchor and instance loss
            # instance loss and anchor loss
            flags = self.flag[index]
            instance_indices = index[flags < 0]
            anchor_indices = index[flags >= 0]
            if local_neighbor_indices is not None and local_neighbor_indices.dim() == 1:
                local_neighbor_indices = local_neighbor_indices.unsqueeze(1)

            # non-parametric classification using memory bank
            zn = F.normalize(zp, p=2, dim=1)
            # For each image get similarity with neighbour
            if self.autograd.requires_grad:
                sim = MemoryBank.apply(zn, index, self.memory, self.params)
            else:
                sim = MemoryBank.inf(zn, self.memory, self.params)
            pred = F.softmax(sim, dim=1)

            batch_size = zp.size(0)

            if len(instance_indices) == 0:
                instance_loss = torch.tensor(0, dtype=torch.float32, device=zp.device)
            else:
                pred_instance = pred[flags < 0, instance_indices]
                if local_neighbor_indices is not None:
                    # calculate local neighbor similarity
                    local_instance_neighbor_indices = local_neighbor_indices[flags < 0]
                    assert torch.all((instance_indices.unsqueeze(1) - local_instance_neighbor_indices) != 0)
                    pred_local = torch.gather(pred[flags < 0].unsqueeze(1), 2,
                                              local_instance_neighbor_indices.unsqueeze(1)).squeeze(1)

                    instance_loss = -torch.log(pred_instance + pred_local.sum(dim=1) + self.const).sum() / batch_size
                else:
                    instance_loss = -torch.log(pred_instance + self.const).sum() / batch_size

            if len(anchor_indices) == 0:
                anchor_loss = torch.tensor(0, dtype=torch.float32, device=zp.device)
            else:
                anchor_neighbor_indices = self.neighbors[anchor_indices]  # global index
                global_anchor_indices = torch.cat((anchor_indices.unsqueeze(1), anchor_neighbor_indices), dim=1)
                pred_global = torch.gather(pred[flags >= 0].unsqueeze(1), 2,
                                           global_anchor_indices.unsqueeze(1)).squeeze(1)
                if local_neighbor_indices is not None:
                    local_anchor_neighbor_indices = local_neighbor_indices[flags >= 0]
                    # Remove local neighbor
                    mask = torch.zeros_like(global_anchor_indices, dtype=torch.bool)
                    for i in range(global_anchor_indices.size(0)):
                        mask[i] = torch.isin(global_anchor_indices[i], local_anchor_neighbor_indices[i])
                    pred_global[mask] = 0
                    # Use torch.gather to correctly index pred for neighbors
                    pred_local = torch.gather(pred[flags >= 0].unsqueeze(1), 2,
                                              local_anchor_neighbor_indices.unsqueeze(1)).squeeze(1)
                    anchor_loss = -torch.log(
                        pred_global.sum(dim=1) + pred_local.sum(dim=1) + self.const).sum() / batch_size
                else:
                    anchor_loss = -torch.log(pred_global.sum(dim=1) + self.const).sum() / batch_size

            loss_dt = {
                'instance_loss': instance_loss,
                'anchor_loss': anchor_loss
            }
            return loss_dt

        else:  # validation and test sample don't need to calculate loss, so only return similar training sample's index
            assert not self.training, 'MemoryCluster should be in eval mode'
            zn = F.normalize(zp, p=2, dim=1)
            sim = MemoryBank.inf(zn, self.memory, self.params)
            pred = F.softmax(sim, dim=1)

            # similar training sample's index and similar training sample's flag
            indices = torch.argmax(pred, dim=1)  # training sample in memory
            flags = self.flag[indices]  # anchor or instance of the query sample

            ratio = torch.sum(flags >= 0).float() / flags.size(0)
            loss_dt = {
                'instance_ratio': 1 - ratio,
                'anchor_ratio': ratio
            }
            return loss_dt

    def get_all_cluster(self):
        if self.kmeans.centroids is None:
            self.kmeans.fit_predict(self.memory).reshape(-1)
            self.centroids = self.kmeans.centroids
        else:
            labels = self.kmeans.predict(self.memory)
            return labels
        # memory = self.memory.cpu().detach().numpy()
        # labels = torch.tensor(self.spectral.fit_predict(memory).reshape(-1), device=self.memory.device)
        # self.centroids = torch.stack([self.memory[labels == i].mean(axis=0) for i in range(self.n_cluster)])
        # self.centroids = F.normalize(self.centroids, p=2, dim=1)

    def get_cluster(self, zp):
        if self.kmeans.centroids is None:
            self.kmeans.centroids = self.centroids
        zp = F.normalize(zp, p=2, dim=1)
        labels = self.kmeans.predict(zp)
        return torch.tensor(labels, device=zp.device).reshape(-1)
        # if self.centroids is None:
        #     self.get_all_cluster()
        # zp = F.normalize(zp, p=2, dim=1)
        # similarities = F.cosine_similarity(zp.unsqueeze(1), self.centroids.unsqueeze(0), dim=2)
        # res = torch.argmax(similarities, dim=1)
        # return res

    def visualize_cluster(self, npc_x, label, save_path, random_state=42):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import umap
        import torch.nn.functional as F
        import os
        import matplotlib.pyplot as plt
        import pandas as pd

        npc_x = F.normalize(npc_x, p=2, dim=1)
        cluster = self.get_cluster(npc_x)
        npc_x = npc_x.cpu().detach().numpy()
        cluster = cluster.cpu().detach().numpy()
        label = label.cpu().detach().numpy()  # 将 label 转换为 NumPy 数组

        # 创建 DataFrame 用于后续处理
        cluster_df = pd.DataFrame({
            'cluster': cluster.reshape(-1) if cluster.ndim > 1 else cluster,
            'label': label.reshape(-1) if label.ndim > 1 else label
        })

        # 计算每个 cluster 中 label=1 的比例
        label_1_proportion = (
            cluster_df[cluster_df['label'] == 1]
            .groupby('cluster')
            .size()
            .div(cluster_df.groupby('cluster').size())
            .reset_index(name='proportion')
        )

        label_1_proportion = label_1_proportion.sort_values(by='proportion', ascending=False).reset_index(drop=True)

        # 定义三组 cluster
        high_risk_clusters = []
        medium_risk_clusters = []
        low_risk_clusters = []

        for idx, row in label_1_proportion.iterrows():
            cluster_id = int(row['cluster'])
            if idx < 2:  # 前两个数量最多的 cluster 为高风险
                high_risk_clusters.append(cluster_id)
            elif idx < 4:  # 接下来的两个数量次多的 cluster 为中风险
                medium_risk_clusters.append(cluster_id)
            else:  # 剩下的为低风险
                low_risk_clusters.append(cluster_id)

        # 将每个样本所属的风险组保存到 cluster_df 中
        def get_risk_group(cluster_id):
            if cluster_id in high_risk_clusters:
                return 'High Risk'
            elif cluster_id in medium_risk_clusters:
                return 'Medium Risk'
            else:
                return 'Low Risk'

        cluster_df['risk_group'] = cluster_df['cluster'].apply(get_risk_group)

        # 使用 PCA 降维到 2D 和 3D
        pca = PCA(n_components=2)
        memory_2d_pca = pca.fit_transform(npc_x)
        pca = PCA(n_components=3)
        memory_3d_pca = pca.fit_transform(npc_x)

        # 使用 t-SNE 降维到 2D 和 3D
        tsne = TSNE(n_components=2, random_state=random_state, metric='cosine')
        memory_2d_tsne = tsne.fit_transform(npc_x)
        tsne = TSNE(n_components=3, random_state=random_state, metric='cosine')
        memory_3d_tsne = tsne.fit_transform(npc_x)

        # 使用 UMAP 降维到 2D 和 3D
        reducer = umap.UMAP(n_components=2, random_state=random_state, metric='cosine')
        memory_2d_umap = reducer.fit_transform(npc_x)
        reducer = umap.UMAP(n_components=3, random_state=random_state, metric='cosine')
        memory_3d_umap = reducer.fit_transform(npc_x)

        # 绘制高中低风险组的 2D 和 3D 图形
        def plot_risk_groups(memory_2d, memory_3d, risk_groups, title_prefix, save_prefix):
            # 2D 可视化
            plt.figure(figsize=(8, 6))
            for risk_group in ['High Risk', 'Medium Risk', 'Low Risk']:
                group_indices = risk_groups == risk_group
                plt.scatter(memory_2d[group_indices, 0], memory_2d[group_indices, 1],
                            label=f'{risk_group}', marker='o')

            # 为每个数据点添加 label 数字
            for i in range(len(label)):
                plt.text(memory_2d[i, 0], memory_2d[i, 1], str(label[i]),
                         fontsize=8, ha='right')

            plt.legend()
            plt.title(f'{title_prefix} Visualization (2D)')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.savefig(os.path.join(save_path, f'{save_prefix}_2d.png'))
            plt.close()

            # 3D 可视化
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            for risk_group in ['High Risk', 'Medium Risk', 'Low Risk']:
                group_indices = risk_groups == risk_group
                ax.scatter(memory_3d[group_indices, 0], memory_3d[group_indices, 1],
                           memory_3d[group_indices, 2],
                           label=f'{risk_group}', marker='o')

            ax.legend()
            ax.set_title(f'{title_prefix} Visualization (3D)')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plt.savefig(os.path.join(save_path, f'{save_prefix}_3d.png'))
            plt.close()

        # 绘制风险组的可视化结果
        plot_risk_groups(memory_2d_pca, memory_3d_pca, cluster_df['risk_group'], 'Risk Groups with PCA',
                         'risk_groups_pca')
        plot_risk_groups(memory_2d_tsne, memory_3d_tsne, cluster_df['risk_group'], 'Risk Groups with t-SNE',
                         'risk_groups_tsne')
        plot_risk_groups(memory_2d_umap, memory_3d_umap, cluster_df['risk_group'], 'Risk Groups with UMAP',
                         'risk_groups_umap')

        return cluster


class Stage1Model(nn.Module):
    def __init__(self,
                 n_samples,
                 img_size=64,
                 neighbors=5,
                 n_cluster=10,
                 temperature=0.07,
                 momentum=0.5,
                 const=1e-8,
                 encoder_in_channels=1,
                 encoder_out_channels=1,
                 hidden_dim=512,
                 npc_dim=128,
                 perceptual_loss_ratio=0,
                 activation='ReLU'):
        super().__init__()
        self.encoder_in_channels = encoder_in_channels
        # stage 1
        self.unet = UNet64(in_channels=encoder_in_channels, out_channels=encoder_out_channels)
        pool_size = img_size // 8  # unet64 downsample 3 times
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.fc = nn.Linear(hidden_dim, npc_dim)
        self.act = getattr(nn, activation)()
        self.mc = MemoryCluster(n_samples, npc_dim, neighbors, n_cluster, temperature, momentum, const)
        self.recon_loss = nn.MSELoss(reduction='mean')
        self.perceptual_loss = PerceptualLoss()
        self.perceptual_loss_ratio = perceptual_loss_ratio

    def forward(self, x, index, local_neighbor_indices, loss=True):
        assert x.size(1) == self.encoder_in_channels, (f'input channel should be {self.encoder_in_channels},'
                                                       f' but got {x.size(1)}')

        hid_x, x_hat = self.unet(x)
        hid_x = self.pool(hid_x).squeeze()
        npc_x = self.act(self.fc(hid_x))
        if not loss:
            cluster = self.mc.get_cluster(npc_x)
            return cluster, hid_x
        loss_dt = self.mc(npc_x, index, local_neighbor_indices)
        # used in old version(20250412)
        # recon = self.recon_loss(x_hat, x)
        if self.perceptual_loss_ratio > 0:
            recon = self.recon_loss(x_hat, x) + self.perceptual_loss(x_hat, x) * self.perceptual_loss_ratio
        else:
            recon = self.recon_loss(x_hat, x)
        loss_dt.update({'recon_loss': recon})
        return loss_dt

    def update_anchor(self, search_rate):
        self.mc.update_anchor(search_rate)

    def get_all_cluster(self):
        return self.mc.get_all_cluster()

    def visualize_cluster(self, x, label=None, save_path=None):
        assert x.size(-1) == 64 or x.size(-2) == 64, 'input size should be 64x64'
        assert x.size(1) == self.encoder_in_channels, (f'input channel should be {self.encoder_in_channels},'
                                                       f' but got {x.size(1)}')
        hid_x, x_hat = self.unet(x)
        hid_x = self.pool(hid_x).squeeze()
        npc_x = self.act(self.fc(hid_x))
        if save_path is not None:
            return self.mc.visualize_cluster(npc_x, label, save_path)
        else:
            return self.mc.get_cluster(npc_x).cpu().detach().numpy()


class Stage1ModelLoss(nn.Module):
    def __init__(self, recon_ratio=1):
        super().__init__()
        self.recon_ratio = recon_ratio

    def forward(self, loss_dt):
        if self.training:
            loss = loss_dt['instance_loss'] + loss_dt['anchor_loss'] + loss_dt['recon_loss'] * self.recon_ratio
            # loss = auto_scaled_loss([loss_dt['instance_loss'] + loss_dt['anchor_loss'], loss_dt['recon_loss']],)
        else:
            # use for val_loss to prevent early stop
            loss = loss_dt['recon_loss']
        loss_dt.update({'loss': loss})
        return loss_dt


class Stage2Model(nn.Module):
    def __init__(self,
                 n_cluster=10,
                 seq_len=3,
                 num_classes=3,
                 transformer_layers=6,
                 hidden_dim=512,
                 npc_dim=128,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='GELU',
                 attention_dropout=0.1,
                 drop_path_rate=0.1,
                 group_q=False,
                 group_k=True,
                 softmax_temp=1,
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes  # used in pl module

        # stage 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + npc_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.position = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        # self.register_buffer('position', sinusoidal_embedding(n_channels=seq_len, dim=hidden_dim))
        self.transformer = AcTransformerEncoder(d_model=hidden_dim, num_heads=num_heads, number_clusters=n_cluster,
                                                dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                                                attention_dropout=attention_dropout, drop_path_rate=drop_path_rate,
                                                group_q=group_q, group_k=group_k, softmax_temp=softmax_temp,
                                                n_layers=transformer_layers)
        self.cls_head = ViTSequentialClassificationHead(hidden_dim, num_classes)

    def forward(self, hid_x, clusters, meaning=False):
        # indices, _ = self.mc.get_anchor(npc_x)  # search anchor in memory
        # stored_npc_x = self.mc.memory[indices]
        # stored_npc_x = stored_npc_x.reshape(-1, self.seq_len, stored_npc_x.size(-1))

        hid_x = self.fc2(hid_x)
        hid_x = hid_x.reshape(-1, self.seq_len, hid_x.size(-1))  # reshape to seq_len
        clusters = clusters.reshape(-1, self.seq_len)
        # hid_x = self.fc3(torch.cat((hid_x, stored_npc_x), dim=-1))
        # cls_token = self.cls_token.expand(hid_x.size(0), -1, -1)
        # embedding = torch.cat((cls_token, hid_x), dim=1)
        embedding = hid_x
        # embedding = embedding.clone()  # used in shap
        embedding += self.position
        sim = self.transformer(embedding, clusters)
        pred_cls = self.cls_head(sim)
        if meaning:
            pred_cls = pred_cls.reshape(-1, self.seq_len, pred_cls.size(-1))
            pred_cls = torch.mean(pred_cls, dim=1)
        else:
            pred_cls = pred_cls.reshape(-1, pred_cls.size(-1))

        return pred_cls


class Stage2ModelLoss(nn.Module):
    def __init__(self, num_classes=3, cls_class_ratio=1/3):
        super().__init__()
        self.num_classes = num_classes
        if isinstance(cls_class_ratio, str):
            cls_class_ratio = eval(cls_class_ratio)
        if num_classes == 3:
            self.cls_loss = nn.CrossEntropyLoss()
        else:  # 2
            self.register_buffer('cls_weight', torch.tensor([1 - cls_class_ratio, cls_class_ratio], dtype=torch.float32))
            self.cls_loss = nn.CrossEntropyLoss(weight=self.cls_weight)

    def forward(self, pred_cls, cls):
        if cls.ndim == 2:
            pred_cls = F.softmax(pred_cls, dim=1)
            loss = -cls * torch.log(pred_cls + 1e-9)
            if self.num_classes == 2:
                loss = torch.sum(loss * self.cls_weight, dim=1)
            else:
                loss = torch.sum(loss, dim=1)
            loss = torch.mean(loss)
        else:
            loss = self.cls_loss(pred_cls, cls)
        return loss


# class Stage2ModelLoss(nn.Module):
#     def __init__(self, num_classes=3):
#         super().__init__()
#         self.gamma = 2.0
#         self.epsilon = 1e-9
#         self.num_classes = num_classes
#         if num_classes == 3:
#             self.cls_loss = nn.CrossEntropyLoss()
#         else:  # 2
#             self.register_buffer('cls_weight', torch.tensor([2 / 3, 1 / 3], dtype=torch.float32))
#             self.cls_loss = nn.CrossEntropyLoss(weight=self.cls_weight)

#     def forward(self, pred_cls, cls):
#         if cls.ndim == 1:
#             cls = cls.unsqueeze(1)  # (b, 1)
#             cls = torch.zeros_like(pred_cls).scatter_(1, cls, 1)  # (b, 2)
#         pred_cls = F.softmax(pred_cls, dim=1)
#         p_t = cls * pred_cls + (1 - cls) * (1 - pred_cls)
#         loss = -cls * torch.pow(1 - p_t, self.gamma) * torch.log(pred_cls + self.epsilon)
#         if self.num_classes == 2:
#             loss = torch.sum(loss * self.cls_weight, dim=1)
#         else:
#             loss = torch.sum(loss, dim=1)
#         loss = torch.mean(loss)
#         return loss


if __name__ == "__main__":
    from thop import profile, clever_format

    # test model
    model = Stage1Model(n_samples=1000, encoder_in_channels=1, encoder_out_channels=1, perceptual_loss_ratio=0.1)
    x = torch.randn(10, 1, 64, 64)
    index = torch.randint(0, 1000, (10,))
    local_neighbor_indices = torch.randint(0, 1000, (10, 5))

    loss_dt = model(x, index, local_neighbor_indices)
    print(loss_dt)
    model.get_all_cluster()

    flops, params = profile(model, inputs=(x, index, local_neighbor_indices))      # 返回纯数字
    print(clever_format([flops, params], "%.3f"))

    model2 = Stage2Model(n_cluster=10, seq_len=3, num_classes=2)
    hid_x = torch.randn(9, 512)
    clusters = torch.randint(0, 10, (9,))
    pred_cls = model2(hid_x, clusters)
    print(pred_cls.shape)

    flops, params = profile(model2, inputs=(hid_x, clusters))      # 返回纯数字
    print(clever_format([flops, params], "%.3f"))

    criterion2 = Stage2ModelLoss(num_classes=2)
    cls = torch.randint(0, 2, (9,))
    loss2 = criterion2(pred_cls, cls)
    print(loss2)