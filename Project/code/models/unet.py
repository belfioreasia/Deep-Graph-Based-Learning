import torch.nn.functional as F
import torch.nn as nn
import torch


class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip[:, :, :x.shape[2], :x.shape[3]]), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

################################## NEW ADDITIONS ##########################################

class DenseGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(DenseGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = nn.ReLU()

    def forward(self, X, A):
        """
        Args:
            X: Node features, shape [B, N, in_features]
            A: Adjacency matrix, shape [B, N, N] (weighted, values between 0 and 1)
        Returns:
            Activated features, shape [B, N, out_features]
        """
        B, N, _ = X.shape
        # Add self-loops: A_hat = A + I
        I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, N, N)
        A_hat = A + I
        # Compute degree: D = sum(A_hat, dim=-1)
        D = torch.sum(A_hat, dim=-1)
        # Compute D^{-1/2}
        D_inv_sqrt = torch.diag_embed(torch.pow(D + 1e-6, -0.5))
        # Normalize adjacency: A_norm = D^{-1/2} A_hat D^{-1/2}
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt, A_hat), D_inv_sqrt)
        # Multiply normalized adjacency with node features.
        support = torch.bmm(A_norm, X)  # [B, N, in_features]
        out = self.linear(support)
        return self.activation(out)


class ContextEmbedGNN(nn.Module):
    def __init__(self, num_nodes, hidden_features, out_features):
        """
        Args:
            num_nodes: Number of nodes in the context graph (e.g., 160).
            hidden_features: Intermediate feature dimension.
            out_features: Final output dimension (e.g., 2*n_feat or n_feat).
        """
        super(ContextEmbedGNN, self).__init__()
        self.num_nodes = num_nodes
        # Define learnable initial node features. We initialize them as an identity.
        self.X0 = nn.Parameter(torch.eye(num_nodes))
        # Two DenseGCN layers.
        self.gcn1 = DenseGCNLayer(in_features=num_nodes, out_features=hidden_features)
        self.gcn2 = DenseGCNLayer(in_features=hidden_features, out_features=out_features)
        # Flatten node features and map to a global embedding.
        self.linear = nn.Linear(num_nodes * out_features, out_features)

    def forward(self, A_context):
        """
        Args:
            A_context: Context adjacency matrix, shape [B, num_nodes, num_nodes].
        Returns:
            embedding: Global context embedding, shape [B, out_features, 1, 1].
        """
        B, N, _ = A_context.shape  # N should equal num_nodes
        # Expand learnable node features to batch size.
        X0 = self.X0.unsqueeze(0).expand(B, -1, -1)  # [B, num_nodes, num_nodes]
        # Force A_context to be positive (using softplus).
        A_context = F.softplus(A_context)
        # Process through the two GCN layers.
        h = self.gcn1(X0, A_context)  # [B, num_nodes, hidden_features]
        h = self.gcn2(h, A_context)   # [B, num_nodes, out_features]
        # Flatten per-node features.
        h_flat = h.view(B, -1)        # [B, num_nodes * out_features]
        # Map to global embedding.
        embedding = self.linear(h_flat)  # [B, out_features]
        return embedding.view(B, -1, 1, 1)

################################## NEW ADDITIONS ##########################################

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=160**2):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(67), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 67, 67),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c_batch = c.shape[0]
        c = c.view(c_batch, -1)
        c_feats = c.shape[1]

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, c_feats)
        context_mask = 1 - context_mask  # need to flip 0 <-> 1
        c = c * context_mask

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


class ContextUnetGraph(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_context=160):
        """
        Args:
            in_channels: Number of channels of the noisy image (e.g., 1).
            n_feat: Base feature dimension.
            n_context: Number of nodes in the context graph (e.g., 160).
        Assumptions:
            - Noisy image x: [B, 1, 268, 268] → graph with 268 nodes.
            - Context c: [B, n_context, n_context] (as an adjacency matrix for a graph with n_context nodes).
        """
        super(ContextUnetGraph, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        # Keep the original convolutional branch for x.
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(67), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        # Replace MLP context embedder with GCN-based ones.
        self.contextembed1 = ContextEmbedGNN(num_nodes=n_context, hidden_features=n_feat, out_features=2 * n_feat)
        self.contextembed2 = ContextEmbedGNN(num_nodes=n_context, hidden_features=n_feat, out_features=1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 67, 67),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        """
        Args:
            x: Noisy image, [B, 1, 268, 268].
            c: Context, [B, n_classes, n_classes].
            t: Timestep, [B].
            context_mask: [B] (binary; 1 means drop context).
        Returns:
            out: Reconstructed image, [B, 1, 268, 268].
        """
        B = x.size(0)
        # Process x with the convolutional branch (keeping x in image form).
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        # Note: Use the output of the convolutional branch as is.
        hiddenvec = self.to_vec(down2)  # [B, 2*n_feat, H, W] (H and W depend on pooling)
        
        c = c.view(B, 160, 160)
        # Process context using the new GCN-based context embedders.
        c = c * (1 - context_mask.view(B, 1, 1))
        cemb1 = self.contextembed1(c)  # [B, 2*n_feat, 1, 1]
        cemb2 = self.contextembed2(c)  # [B, 1*n_feat, 1, 1]

        # Process time embeddings.
        t = t.view(B, 1)
        temb1 = self.timeembed1(t).view(B, 2 * self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(B, self.n_feat, 1, 1)

        # Combine context and time embeddings.
        combined_global1 = cemb1   # [B, 2*n_feat, 1, 1]
        combined_global2 = cemb2   # [B, n_feat, 1, 1]

        # Now, add the global combined embeddings to the feature maps from x.
        # hiddenvec is from the conv branch and is a 4D tensor.
        combined = hiddenvec + combined_global1  # broadcast over spatial dimensions

        # Proceed with the up-convolutions.
        up1 = self.up0(combined)  # expected to be [B, 2*n_feat, H', W']
        up2 = self.up1(combined_global1 * up1 + temb1, down2)  # using skip connection from down2
        up3 = self.up2(combined_global2 * up2 + temb2, down1)  # using skip connection from down1
        out = self.out(torch.cat((up3, x), 1))
        return out
