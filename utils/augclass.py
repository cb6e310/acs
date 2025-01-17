import numpy as np
import tsaug
import torch
import time
from torch.nn.functional import interpolate

import braindecode.augmentation as bdaug


def totensor(x):
    return torch.from_numpy(x).type(torch.FloatTensor).cuda()


class Normalize:
    def __call__(self, batch):
        mean = batch.mean(dim=(1, 2), keepdim=True)
        std = batch.std(dim=(1, 2), keepdim=True)
        return (batch - mean) / std


class crop:
    def __init__(self, resize, crop_size=0.8) -> None:
        crop_size = int(resize * crop_size)
        self.aug = tsaug.Crop(size=crop_size, resize=resize)

    def __call__(self, x):
        x = x.cpu().detach().numpy()
        x = self.aug.augment(x)
        return totensor(x.astype(np.float32))


class jitter:
    def __init__(self, sigma=0.3, random_sigma=False, frac=None) -> None:
        self.sigma = sigma
        self.random_sigma = random_sigma
        self.frac = frac

    def __call__(self, x):
        if self.random_sigma:
            # Random sigma for each sample in the batch
            sigmas_labels = np.random.uniform(
                0, self.sigma, size=(x.shape[0], 1, 1)
            )  # (batch_size, 1, 1)
            sigmas = torch.tensor(
                sigmas_labels, dtype=torch.float32, device=x.device
            ).expand(-1, x.shape[1], x.shape[2])

            # Generate noise
            noise = torch.randn_like(x)

            # Apply noise scaled by sigmas
            pertubated = x + noise * sigmas
            sigmas_labels = torch.tensor(
                sigmas_labels, dtype=torch.float32, device=x.device
            ).squeeze(-1)
            return pertubated, sigmas_labels
        if self.frac:
            sigma_list = [0, 0.1, 0.3, 0.5]
            splits=list(torch.chunk(x, len(sigma_list), dim=0))
            labels = torch.zeros(x.shape[0])

            split_labels = list(torch.chunk(labels, len(sigma_list)))
            for i, split in enumerate(splits):
                current_labels=torch.full_like(split_labels[i], sigma_list[i])
                split_labels[i] = current_labels
                noise=torch.randn_like(split)
                splits[i] = split + noise * sigma_list[i]
            return torch.cat(splits, 0), torch.cat(split_labels, 0)

        else:
            return x + torch.normal(
                mean=0.0, std=self.sigma, size=x.shape, device=x.device
            )


class scaling:
    def __init__(self, max_sigma=0.5, random=False) -> None:
        self.max_sigma = max_sigma
        self.random = random

    def __call__(self, x):
        # random scaling for each sample in the batch from 1-max_sigma to 1+max_sigma
        factor = (torch.rand(x.shape[0], device=x.device) * 2 - 1) * self.max_sigma + 1
        # res = torch.multiply(x, torch.unsqueeze(factor, 1))
        res = x * factor.unsqueeze(1).unsqueeze(1)
        if self.random:
            return res, factor
        else:
            return res


class timeshift:
    def __init__(self, shift_max=10):
        self.shift_max = shift_max

    def __call__(self, x):
        shift = np.random.randint(-self.shift_max, self.shift_max)
        return torch.roll(x, shifts=shift, dims=1)


class time_warp:
    def __init__(self, n_speed_change=100, max_speed_ratio=10) -> None:
        self.transform = tsaug.TimeWarp(
            n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio
        )

    def __call__(self, x_torch):
        x = x_torch.cpu().detach().numpy()
        x_tran = self.transform.augment(x)
        return totensor(x_tran.astype(np.float32))


class magnitude_warp:

    def __init__(self, n_speed_change: int = 100, max_speed_ratio=10) -> None:
        self.transform = tsaug.TimeWarp(
            n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio
        )

    def __call__(self, x_torch):
        x = x_torch.cpu().detach().numpy()
        # x_t = np.transpose(x, (0, 2, 1))
        x_tran = self.transform.augment(x)
        return totensor(x_tran.astype(np.float32))


class window_slice:
    def __init__(self, reduce_ratio=0.5, diff_len=True) -> None:
        self.reduce_ratio = reduce_ratio
        self.diff_len = diff_len

    def __call__(self, x):

        # begin = time.time()
        # x = torch.transpose(x, 2, 1)

        target_len = np.ceil(self.reduce_ratio * x.shape[2]).astype(int)
        if target_len >= x.shape[2]:
            return x
        if self.diff_len:
            starts = np.random.randint(
                low=0, high=x.shape[2] - target_len, size=(x.shape[0])
            ).astype(int)
            ends = (target_len + starts).astype(int)
            croped_x = torch.stack(
                [x[i, :, starts[i] : ends[i]] for i in range(x.shape[0])], 0
            )

        else:
            start = np.random.randint(low=0, high=x.shape[2] - target_len)
            end = target_len + start
            croped_x = x[:, :, start:end]

        ret = interpolate(croped_x, x.shape[2], mode="linear", align_corners=False)
        ret = torch.transpose(ret, 2, 1)
        # end = time.time()
        # old_window_slice()(x)
        # end2 = time.time()
        # print(end-begin,end2-end)
        return ret


class window_warp:
    def __init__(self, window_ratio=0.3, scales=[0.5, 2.0]) -> None:
        self.window_ratio = window_ratio
        self.scales = scales

    def __call__(self, x):
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        x = x.cpu().detach().numpy()
        warp_scales = np.random.choice(self.scales, x.shape[0])
        warp_size = np.ceil(self.window_ratio * x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(
            low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])
        ).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                start_seg = pat[: window_starts[i], dim]
                window_seg = np.interp(
                    np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])),
                    window_steps,
                    pat[window_starts[i] : window_ends[i], dim],
                )
                end_seg = pat[window_ends[i] :, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                ret[i, :, dim] = np.interp(
                    np.arange(x.shape[1]),
                    np.linspace(0, x.shape[1] - 1.0, num=warped.size),
                    warped,
                ).T

        return torch.from_numpy(ret).type(torch.FloatTensor).cuda()


class TimeReverse:
    def __init__(self, random=False) -> None:
        self.aug = bdaug.TimeReverse(probability=1)
        self.random = random

    def __call__(self, x):
        # select half of the samples in batch to reverse
        applied_batch_idx = np.random.choice(x.shape[0], x.shape[0] // 2, replace=False)
        x[applied_batch_idx] = self.aug(x[applied_batch_idx], None)
        # labels:0 for not reversed, 1 for reversed
        labels = torch.zeros(x.shape[0], dtype=torch.long)
        labels[applied_batch_idx] = 1
        labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
        if self.random:
            return x, labels
        else:
            return x


class SignFlip:
    def __init__(self, random) -> None:
        self.aug = bdaug.SignFlip(probability=1)
        self.random = random

    def __call__(self, x):
        # select half of the samples in batch to flip
        applied_batch_idx = np.random.choice(x.shape[0], x.shape[0] // 2, replace=False)
        x[applied_batch_idx] = self.aug(x[applied_batch_idx], None)
        # labels:0 for not flipped, 1 for flipped
        labels = torch.zeros(x.shape[0], dtype=torch.long)
        labels[applied_batch_idx] = 1
        # one-hot encoding
        labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
        if self.random:
            return x, labels
        else:
            return x


class FTSurrogate:
    def __init__(self, p=1, fine_factor=10) -> None:
        self.aug = bdaug.FTSurrogate(p)
        self.fine_factor = fine_factor

    def __call__(self, x):
        fine_factor = self.fine_factor
        # split x into fine_factor numbers of batches
        fined_size = x.shape[0] // fine_factor
        fined_x = list(torch.split(x, fined_size, dim=0))
        noise = torch.rand(fine_factor, 1)
        labels = torch.zeros(x.shape[0], dtype=torch.long)
        labels = list(torch.split(labels, fined_size, dim=0))
        for i in np.arange(len(fined_x)):
            fined_x[i] = self.aug.operation(fined_x[i], None, noise[i], False)[0]
            labels[i] = torch.full(labels[i].size(), noise[i][0])
        # expand noise to the same shape as x
        fined_x = torch.cat(fined_x, 0)
        labels = torch.cat(labels, 0)
        return fined_x, labels


class TimeShift:
    def __init__(self, max_shift=0.1, random=False):
        self.max_shift = max_shift
        self.random = random

    def __call__(self, x):
        batch_size, length, channels = x.size()
        max_shift = int(self.max_shift * length)
        roll_lengths = torch.randint(-max_shift, max_shift + 1, (batch_size,))

        # Create an index tensor for rolling
        indices = (torch.arange(length).unsqueeze(0) - roll_lengths.unsqueeze(1)) % length
        indices = indices.unsqueeze(1).expand(batch_size, channels, -1)
        indices = torch.transpose(indices, 1, 2)
        # Apply the roll using advanced indexing
        rolled_batch = torch.gather(x, 1, indices.cuda())

        # turn roll_lengths to proportion
        roll_lengths = roll_lengths / length
        if self.random:
            return rolled_batch, roll_lengths
        else:
            return rolled_batch


class FrequencyShift:
    def __init__(self, sfreq, max_shift=10, fined_factor=10):
        self.sfreq = sfreq
        self.max_shift = max_shift
        self.aug = bdaug.FrequencyShift(probability=1, sfreq=sfreq)
        self.fined_factor = fined_factor

    def __call__(self, x):
        fined_size = x.shape[0] // self.fined_factor + 1
        freq_shift = np.random.uniform(-self.max_shift, self.max_shift, size=fined_size)
        freq_shift = freq_shift.reshape(-1, 1)
        # split x into fine_factor numbers of batches and to cpu
        fined_x = list(torch.split(x.cpu(), fined_size, dim=0))
        labels = torch.zeros(x.shape[0], dtype=torch.long)
        labels = list(torch.split(labels, fined_size, dim=0))
        for i in np.arange(len(fined_x)):
            fined_x[i] = self.aug.operation(
                fined_x[i], None, freq_shift[i], sfreq=self.sfreq
            )[0]
            labels[i] = torch.full(labels[i].size(), freq_shift[i][0])

        fined_x = torch.cat(fined_x, 0)
        labels = torch.cat(labels, 0)

        return fined_x.cuda(), labels


class Drift:
    def __init__(self, max_drift=0.2, max_drift_points=10, fined_factor=10):
        self.max_drift = max_drift
        self.max_drift_points = max_drift_points
        self.fined_factor = fined_factor

    def __call__(self, x):
        labels = torch.zeros(x.shape[0], dtype=torch.long)
        fined_size = x.shape[0] // self.fined_factor + 1
        # drift_points = torch.randint(1, self.max_drift_points + 1, (self.fined_factor,))
        # drift_points = drift_points.reshape(-1, 1)
        labels = list(torch.split(labels, fined_size, dim=0))
        # split x into fine_factor numbers of batches
        fined_x = list(torch.split(x.cpu(), fined_size, dim=0))
        # drift_points = torch.randint(1, self.max_drift_points + 1, (len(fined_x),))
        drift_points = torch.range(1, self.max_drift_points + 1)
        max_drifts = torch.rand(len(fined_x)) * self.max_drift
        for i in np.arange(len(fined_x)):
            fined_x[i] = torch.from_numpy(
                tsaug.Drift(
                    max_drifts[i].item(),
                    drift_points.int().tolist(),
                    kind="multiplicative",
                ).augment(fined_x[i].numpy())
            )
            labels[i] = torch.full(labels[i].size(), drift_points[i])

        fined_x = torch.cat(fined_x, 0)
        labels = torch.cat(labels, 0)
        return fined_x.cuda(), labels
