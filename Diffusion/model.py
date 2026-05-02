import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torchvision import transforms, datasets
import math


#残差ブロック
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # 埋め込みをこのブロックのチャンネル数に合わせる専用層
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels)
        )
        # チャンネル数が変わる場合、identityを合わせるための1x1畳み込みが必要
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, emb):
        out = self.conv1(F.silu(self.bn1(x)))
        out = out + self.mlp(emb).view(emb.size(0), -1, 1, 1)
        out = self.conv2(F.silu(self.bn2(out)))
        return out + self.shortcut(x)


#ダウンブロック
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.residual = ResidualBlock(in_channels, in_channels, embed_dim)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, emb):
        out1 = self.residual(x, emb)
        out2 = self.residual(out1, emb)
        out3 = F.relu(self.conv(out2))
        return [out1, out2, out3]


#アップブロック
class Upblock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.residual1 = ResidualBlock(in_channels, out_channels, embed_dim)
        self.residual2 = ResidualBlock(2 * out_channels, out_channels, embed_dim)

    def forward(self, x, x_list, emb):
        target_size = x_list[1].shape[2:] 
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = torch.cat((x, x_list[1]), dim=1)
        x = self.residual1(x, emb)
        target_size2 = x_list[0].shape[2:]
        x = F.interpolate(x, size=target_size2, mode='bilinear', align_corners=False)
        x = torch.cat((x, x_list[0]), dim=1)
        x = self.residual2(x, emb)
        return x



#Diffusion U-Net
class UNet(nn.Module):
    def __init__(self, channels, diffusion, embed_dim=32):
        super().__init__()
        self.sinusoidal_embedding = diffusion.sinusoidal_embedding
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.downblock1 = DownBlock(64, 96, embed_dim)
        self.downblock2 = DownBlock(96, 128, embed_dim)
        self.downblock3 = DownBlock(128, 160, embed_dim)
        self.residual = ResidualBlock(160, 160, embed_dim)
        self.upblock1 = Upblock(288, 128, embed_dim)
        self.upblock2 = Upblock(224, 96, embed_dim)
        self.upblock3 = Upblock(160, 64, embed_dim)
        self.conv2 = nn.Conv2d(64, channels, 3, padding=1)

    def forward(self, x, t):
        x = x.float()
        step_embedding = self.sinusoidal_embedding(t, x)
        x = F.relu(self.conv1(x))
        out1 = self.downblock1(x, step_embedding)
        out2 = self.downblock2(out1[2], step_embedding)
        out3 = self.downblock3(out2[2], step_embedding)
        out = self.residual(self.residual(out3[2], step_embedding), step_embedding)
        out = self.upblock1(out, out3, step_embedding)
        out = self.upblock2(out, out2, step_embedding)
        out = self.upblock3(out, out1, step_embedding)
        out = self.conv2(out)

        return out


class Diffusion(nn.Module):
    def __init__(self, loss_fn, device, diffusion_times, size, inverse_times, embedding_dim=32):
        super().__init__()
        self.loss_fn = loss_fn
        self.device = device
        self.diffusion_times = diffusion_times
        #位置埋め込みの次元
        self.embed_dim = embedding_dim
        #逆方向の拡散過程用に、訓練で用いる画像のHWサイズを属性とする
        self.size = size
        #逆方向の拡散過程のステップ数
        self.inverse_times = inverse_times

    #オフセット付きコサイン拡散スケジューリング
    def offset_cosine_diffusion_schedule(self, diffusion_step, x, train=True):
        min_signal_rate = torch.tensor(0.02)
        max_signal_rate = torch.tensor(0.95)
        start_angle = torch.acos(max_signal_rate)
        end_angle = torch.acos(min_signal_rate)

        #diffusion_stepを正規化する
        if train:
            t = diffusion_step.float() / self.diffusion_times
        else:
            t = float(diffusion_step) / self.inverse_times
        diffusion_angles = start_angle + t * (end_angle - start_angle)

        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        noise_rates = noise_rates.view(-1, 1, 1, 1)
        signal_rates = signal_rates.view(-1, 1, 1, 1)
        return noise_rates.to(x), signal_rates.to(x)

    #正弦波埋め込み
    def sinusoidal_embedding(self, t, x):
        half_dim = self.embed_dim // 2

        # 指数スケールで周波数を作成
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]

        # sinとcosを並べて [batch_size, embedding_dim] に
        embeddings = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return embeddings.view(-1, self.embed_dim).to(x)

    #順方向の拡散過程の計算
    def get_diffuse(self, x, t, noise):
        #ノイズの割合を計算する（ここではオフセット付きコサイン拡散スケジューリングを使う）
        noise_rates, signal_rates = self.offset_cosine_diffusion_schedule(t, x)
        x_diffuse = signal_rates * x + noise_rates * noise
        return x_diffuse

    def forward(self, model, imgs):
        #ノイズとステップのサンプリング
        noise = torch.randn_like(imgs)
        t = torch.randint(0, self.diffusion_times, (imgs.size(0),))
        #imgs, t, noiseをGPUに移す
        imgs = imgs.to(device=self.device)
        t = t.to(device=self.device)
        noise = noise.to(device=self.device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16): 
            imgs_diffuse = self.get_diffuse(imgs, t, noise)
            noise_predict = model(imgs_diffuse, t)
            loss = self.loss_fn(noise_predict, noise)

        return loss

    #tステップ目からt-1ステップ目の画像の推定（オフセット付きコサイン拡散スケジューリングを用いる）
    def denoise(self, img, noise_predict, step):
        noise_rates, signal_rates = self.offset_cosine_diffusion_schedule(step, img, train=False)
        noise_prestep, signal_prestep = self.offset_cosine_diffusion_schedule(step-1, img, train=False)
        # signal_rates が 1e-6 未満にならないように強制固定する
        signal_rates = torch.clamp(signal_rates, min=1e-6)
        img_nonoise = (img - noise_rates * noise_predict) / signal_rates
        #t-1ステップからtステップ目のノイズの割合を計算する
        beta_step = 1 - torch.square(signal_rates / signal_prestep)
        #平均の計算(noise_ratesに微小量を足すことによってNanを回避する)
        denom = torch.square(noise_rates) + 1e-8
        mu = (signal_prestep * beta_step * img_nonoise / denom) + (signal_rates * torch.square(noise_prestep) * img / denom)
        #最後のステップのみノイズを加えたくないため、最後のステップかどうかで条件分岐
        if step >= 2:
            #分散の計算
            std = torch.sqrt((noise_prestep / noise_rates) * beta_step)
            #εのサンプリング
            epsilon = torch.randn_like(img)
            #t-1ステップ目の画像の計算
            img_denoise = mu + std * epsilon
        else:
            img_denoise = mu
        return img_denoise.to(img)

    #逆方向の拡散過程で画像を生成する（lengthは読み込む画像と同じサイズを想定）
    def inverse(self, model, channels):
        img_noise = torch.randn(1, channels, self.size, self.size, device=self.device)
        for step in range(self.inverse_times, 0, -1):
            model.eval()
            noise_predict = model(img_noise, torch.tensor([step], device=self.device)).float()
            img_noise = self.denoise(img_noise, noise_predict, step)
        img = img_noise
        # 勾配計算をオフにし、CPUへ移動してNumPy配列に変換
        # サイズが1の次元（バッチとチャンネル）を消して (32, 32) にする
        img_array = img.detach().cpu().squeeze().numpy()       
        return img_array