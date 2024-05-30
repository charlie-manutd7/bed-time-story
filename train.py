import os
import torch
import json
import utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import SynthesizerTrn, MultiPeriodDiscriminator
from data_utils import TextAudioLoader, TextAudioCollate
from text.symbols import symbols
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('configs/ljs_base.json', 'r') as f:
        hps = json.load(f)

    model_dir = hps['model_dir']

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Model directory created: {model_dir}")

    if not os.access(model_dir, os.W_OK):
        raise PermissionError(f"Directory {model_dir} is not writable")
    else:
        print(f"Directory {model_dir} is writable")

    logger = utils.get_logger(model_dir)
    logger.info(hps)
    utils.check_git_hash(model_dir)
    writer = SummaryWriter(log_dir=model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))

    torch.manual_seed(hps['train']['seed'])

    train_dataset = TextAudioLoader(hps['data']['training_files'], hps['data'])
    print(f"Number of training samples: {len(train_dataset)}")
    collate_fn = TextAudioCollate(hps['data']['hop_length'])
    train_loader = DataLoader(train_dataset, num_workers=1, shuffle=True, pin_memory=True,
                              collate_fn=collate_fn, batch_size=hps['train']['batch_size'])
    eval_dataset = TextAudioLoader(
        hps['data']['validation_files'], hps['data'])
    eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False, pin_memory=True,
                             collate_fn=collate_fn, batch_size=hps['train']['batch_size'])
    print(f"Number of validation samples: {len(eval_dataset)}")

    net_g = SynthesizerTrn(len(symbols), hps['data']['filter_length'] // 2 + 1,
                           hps['train']['segment_size'] // hps['data']['hop_length'], **hps['model']).to(device)
    net_d = MultiPeriodDiscriminator(
        hps['model']['use_spectral_norm']).to(device)
    optim_g = torch.optim.AdamW(net_g.parameters(), hps['train']['learning_rate'],
                                betas=hps['train']['betas'], eps=hps['train']['eps'])
    optim_d = torch.optim.AdamW(net_d.parameters(), hps['train']['learning_rate'],
                                betas=hps['train']['betas'], eps=hps['train']['eps'])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(model_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        print(f"Error loading checkpoints: {e}")
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps['train']['lr_decay'], last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps['train']['lr_decay'], last_epoch=epoch_str-2)

    scaler = torch.cuda.amp.GradScaler(enabled=hps['train']['fp16_run'])

    for epoch in range(epoch_str, hps['train']['epochs'] + 1):
        train_and_evaluate(epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [
                           train_loader, eval_loader], logger, [writer, writer_eval], device)
        scheduler_g.step()
        scheduler_d.step()

        try:
            checkpoint_path_g = os.path.join(model_dir, f"G_{global_step}.pth")
            checkpoint_path_d = os.path.join(model_dir, f"D_{global_step}.pth")

            print(
                f"Attempting to save checkpoint for generator to {checkpoint_path_g}...")
            utils.save_checkpoint(
                net_g, optim_g, hps['train']['learning_rate'], epoch, checkpoint_path_g)
            print(f"Checkpoint for generator saved to {checkpoint_path_g}")

            print(
                f"Attempting to save checkpoint for discriminator to {checkpoint_path_d}...")
            utils.save_checkpoint(
                net_d, optim_d, hps['train']['learning_rate'], epoch, checkpoint_path_d)
            print(f"Checkpoint for discriminator saved to {checkpoint_path_d}")

            print(f"Checkpoints saved for epoch {epoch}")
        except Exception as e:
            print(f"Error saving checkpoints: {e}")


def train_and_evaluate(epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, device):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    writer, writer_eval = writers if writers is not None else (None, None)

    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)

        with torch.cuda.amp.autocast(enabled=hps['train']['fp16_run']):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                x, x_lengths, spec, spec_lengths)

            mel = utils.spec_to_mel_torch(spec, hps['data']['filter_length'], hps['data']['n_mel_channels'],
                                          hps['data']['sampling_rate'], hps['data']['mel_fmin'], hps['data']['mel_fmax'])
            y_mel = utils.slice_segments(
                mel, ids_slice, hps['train']['segment_size'] // hps['data']['hop_length'])
            y_hat_mel = utils.mel_spectrogram_torch(y_hat.squeeze(1), hps['data']['filter_length'], hps['data']['n_mel_channels'],
                                                    hps['data']['sampling_rate'], hps['data']['hop_length'], hps['data']['win_length'], hps['data']['mel_fmin'], hps['data']['mel_fmax'])

            y = utils.slice_segments(
                y, ids_slice * hps['data']['hop_length'], hps['train']['segment_size'])

            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with torch.cuda.amp.autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = utils.discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = utils.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with torch.cuda.amp.autocast(enabled=hps['train']['fp16_run']):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with torch.cuda.amp.autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = nn.functional.l1_loss(
                    y_mel, y_hat_mel) * hps['train']['c_mel']
                loss_kl = utils.kl_loss(
                    z_p, logs_q, m_p, logs_p, z_mask) * hps['train']['c_kl']

                loss_fm = utils.feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = utils.generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = utils.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if global_step % hps['train']['log_interval'] == 0:
            lr = optim_g.param_groups[0]['lr']
            losses = [loss_disc, loss_gen, loss_fm,
                      loss_mel, loss_dur, loss_kl]
            logger.info(
                f'Train Epoch: {epoch} [{100. * batch_idx / len(train_loader):.0f}%]')
            logger.info([x.item() for x in losses] + [global_step, lr])


def evaluate(hps, generator, eval_loader, writer_eval, device):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
            x, x_lengths = x.to(device), x_lengths.to(device)
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)

            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            break

        y_hat, attn, mask, * \
            _ = generator.module.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * hps['data']['hop_length']

        mel = utils.spec_to_mel_torch(spec, hps['data']['filter_length'], hps['data']['n_mel_channels'],
                                      hps['data']['sampling_rate'], hps['data']['mel_fmin'], hps['data']['mel_fmax'])
        y_hat_mel = utils.mel_spectrogram_torch(y_hat.squeeze(1).float(), hps['data']['filter_length'], hps['data']['n_mel_channels'],
                                                hps['data']['sampling_rate'], hps['data']['hop_length'], hps['data']['win_length'], hps['data']['mel_fmin'], hps['data']['mel_fmax'])

    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
        "gen/audio": y_hat[0, :, :y_hat_lengths[0]]
    }
    if global_step == 0:
        image_dict.update(
            {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps['data']['sampling_rate']
    )
    generator.train()
