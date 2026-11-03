import shutil
import torch
import torch.optim as optim
from datetime import datetime
from .utils import LogManager

def net_aux_optimizer(model, args):
    net_params = []
    aux_params = []


    net_params += [
        param for name, param in model.named_parameters() if param.requires_grad and not name.endswith(".quantiles")
    ]
    aux_params += [
        param for name, param in model.named_parameters() if param.requires_grad and name.endswith(".quantiles")
    ]
        
    optimizer = optim.AdamW(
        net_params, lr=args.learning_rate
    )
    if len(aux_params) > 0:
        aux_optimizer = optim.AdamW(
            aux_params, lr=args.aux_learning_rate
        )
    else:
        aux_optimizer = None

    return optimizer, aux_optimizer


def load_checkpoints(
        args,
        model,
        optimizer,
        aux_optimizer,
        lr_scheduler,
        log_keys_train,
        log_keys_test,
        len_trainloader
):
    log_dir = args.log_dir

    log_train = LogManager(filepath=log_dir + f"/train.txt")
    log_test = LogManager(filepath=log_dir + f"/test.txt")
    log_etc = open(log_dir + f"/args.txt", "a")

    columns_train, columns_test = f"", f""
    for n, key in enumerate(log_keys_train):
        columns_train += f"{n + 1:02d}. {key}\t"
    for n, key in enumerate(log_keys_test):
        columns_test += f"{n + 1:02d}. {key}\t"
    columns_train = columns_train[:-1] + "\n"
    columns_test = columns_test[:-1] + "\n"

    epoch_latest = -1
    last_epoch = 0
    global_step = 0
    best_epoch = 0
    best_loss = float("inf")

    if args.checkpoint is not None:  # load from previous checkpoint
        ckpt_filelist = args.checkpoint

        for ckpt in ckpt_filelist:
            checkpoint_str = ckpt
            checkpoint = torch.load(checkpoint_str, map_location=args.device)
            if not args.checkpoint_only_weight:
                epoch_latest = checkpoint["epoch"]
            if "state_dict" in checkpoint.keys():
                if args.DDP:
                    try:
                        model.module.load_state_dict(checkpoint["state_dict"], strict=True)
                        print('checkpoint loaded')
                    except:
                        model.module.load_state_dict(state_dict = checkpoint["state_dict"], strict = False)
                        print('does not match')      
                else:
                    try:
                        model.load_state_dict(checkpoint["state_dict"], strict=True)
                        print(key,' loaded')
                    except:
                        model.load_state_dict(state_dict = checkpoint["state_dict"], strict=False)
                        print(key,' does not match')       
                                    
            if args.checkpoint_only_weight:
                log_train.create(columns=columns_train, l_dataloader=len_trainloader)
                log_test.create(columns=columns_test)

            elif epoch_latest != -1:
                print("Loading", checkpoint_str)
                last_epoch = checkpoint["epoch"] + 1
                global_step = checkpoint["global_step"]
                best_loss = checkpoint["best_loss"]
                best_epoch = checkpoint["best_epoch"]

                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    print('loaded optimizer')
                except:
                    pass

                if aux_optimizer is not None and "aux_optimizer" in checkpoint.keys():
                    aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
                if "lr_scheduler" in checkpoint.keys() and lr_scheduler is not None:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

                if torch.distributed.get_rank() == args.gpu:
                    log_train.resume(global_step)
                    log_test.resume(global_step)
                    log_etc.write(f"\n{datetime.now().strftime('%Y%m%d-%H%M%S')}\n"
                                  f"Loading {args.checkpoint}\n")

            del checkpoint
            torch.cuda.empty_cache()

    else:
        log_train.create(columns=columns_train, l_dataloader=len_trainloader)
        log_test.create(columns=columns_test)

    if torch.distributed.get_rank() == args.gpu:
        log_args = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}\n\n<<<<<<< training information >>>>>>>\n\n"
        for k, v in vars(args).items():
            log_args += f"--{k} {v} \\\n"
        log_args += f"\n\n\n"
        log_etc.write(log_args)
        log_etc.flush()
        log_etc.close()

    return model.to(args.device), optimizer, aux_optimizer, lr_scheduler, last_epoch, global_step, best_loss, best_epoch, log_train, log_test


def save_checkpoint(state, is_best, filedir="", filename="checkpoint", suffix=""):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filedir + '/best_loss' + suffix + '.pth.tar')
        