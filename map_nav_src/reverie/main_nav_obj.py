import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init import get_tokenizer

from utils.data import ImageFeaturesDB

from reverie.agent_obj import GMapObjectNavAgent
from reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps
from reverie.env import ReverieObjectNavBatch
from reverie.parser import parse_args


def build_dataset(args, rank=0):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    obj_db = ObjectFeatureDB(args.obj_ft_file, args.obj_feat_size)
    obj2vps = load_obj2vps(os.path.join(args.anno_dir, 'BBoxes.json'))

    dataset_class = ReverieObjectNavBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        aug_env = dataset_class(
            feat_db, obj_db, aug_instr_data, args.connectivity_dir, obj2vps, 
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,
        )
    else:
        aug_env = None

    if args.aug_only:
        train_env, aug_env = aug_env, None
        args.aug = None
    else:
        train_instr_data = construct_instrs(
            args.anno_dir, args.dataset, ['train'], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        train_env = dataset_class(
            feat_db, obj_db, train_instr_data, args.connectivity_dir, obj2vps,
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name='train', 
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,
        )

    # val_env_names = ['val_train_seen']
    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    if args.submit:
        val_env_names.append('test')
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            feat_db, obj_db, val_instr_data, args.connectivity_dir, obj2vps, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            max_objects=None, multi_endpoints=False, multi_startpoints=False,
        )   # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapObjectNavAgent
    listner = agent_class(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)
        # return

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}

    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                listner.train(1, feedback=args.feedback)

                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            OG_loss = sum(listner.logs['OG_loss']) / max(len(listner.logs['OG_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/OG_loss", OG_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # select model by spl
                if env_name in best_val:
                    if score_summary['spl'] >= best_val[env_name]['spl']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
            prefix, env_name, args.fusion))
        if os.path.exists(output_file):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds, open(output_file, 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
                
def apply_sgr(module_list, p=0.05, alpha=-0.2):
    denom = alpha * p + (1.0 - p)
    if abs(denom) < 1e-8:
        raise ValueError("Invalid SGR denominator; choose different p/alpha.")

    for module in module_list:
        for param in module.parameters():
            if param.grad is None:
                continue

            grad = param.grad
            mask = (torch.rand_like(grad) < p)

            grad_selected = alpha * grad
            grad_unselected = grad / denom

            param.grad = torch.where(mask, grad_selected, grad_unselected)

# Test on small sample 

def feedtta_valid(args, train_env, val_envs, rank=-1):
    import os
    import json
    import torch
    from collections import defaultdict

    def _set_trainable_tta_params(model):
        # Freeze everything first
        for _, p in model.named_parameters():
            p.requires_grad = False

        trainable_names = []
        for n, p in model.named_parameters():
            lname = n.lower()

            # Keep cross-modal / navigation-ish blocks trainable.
            # Do NOT open language encoder.
            keep = (
                ("local_encoder.encoder.x_layers" in lname) or
                ("global_encoder" in lname) or
                ("sap_fuse" in lname)
            )

            if keep and ("lang_encoder" not in lname):
                p.requires_grad = True
                trainable_names.append(n)

        return trainable_names

    def _build_tta_optimizer(agent, lr):
        params = [p for p in agent.vln_bert.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable params found for TTA.")
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

    default_gpu = is_default_gpu(args)

    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    target_env_name = getattr(args, 'tta_env_name', 'val_unseen')
    env = val_envs[target_env_name]
    agent.env = env

    tta_steps = int(getattr(args, 'tta_steps', 10))
    tta_lr = float(getattr(args, 'tta_lr', 5e-6))
    grad_clip = float(getattr(args, 'tta_grad_clip', 1.0))
    sgr_p = float(getattr(args, 'sgr_p', 0.05))
    sgr_alpha = float(getattr(args, 'sgr_alpha', -0.2))

    if getattr(args, 'batch_size', None) != 1:
        print("[WARN] FEEDTTA paper-style setup expects --batch_size 1")

    trainable_names = _set_trainable_tta_params(agent.vln_bert)
    optimizer = _build_tta_optimizer(agent, tta_lr)

    if hasattr(agent, 'critic'):
        for p in agent.critic.parameters():
            p.requires_grad = False
        agent.critic.eval()

    print("\n===== FEEDTTA CONFIG =====")
    print("tta_env_name =", target_env_name)
    print("tta_steps    =", tta_steps)
    print("tta_lr       =", tta_lr)
    print("grad_clip    =", grad_clip)
    print("sgr_p        =", sgr_p)
    print("sgr_alpha    =", sgr_alpha)
    print("trainable params in vln_bert =", len(trainable_names))
    # print("first 40 trainable names:")
    # for n in trainable_names[:40]:
    #     print("  ", n)

    if default_gpu:
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, 'feedtta_validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)

    # -------------------------
    # Baseline eval before TTA
    # -------------------------
    print("\n===== BASELINE EVAL BEFORE ADAPTATION =====")
    agent.vln_bert.eval()
    if hasattr(agent, 'critic'):
        agent.critic.eval()

    agent.logs = defaultdict(list)
    agent.test(use_dropout=False, feedback='argmax', iters=None)
    preds_before = agent.get_results(detailed_output=args.detailed_output)
    score_summary_before, _ = env.eval_metrics(preds_before)

    print(
        "[BEFORE] Env name: %s, action_steps: %.2f, steps: %.2f, lengths: %.2f, "
        "sr: %.2f, oracle_sr: %.2f, spl: %.2f, rgs: %.2f, rgspl: %.2f"
        % (
            target_env_name,
            score_summary_before['action_steps'],
            score_summary_before['steps'],
            score_summary_before['lengths'],
            score_summary_before['sr'],
            score_summary_before['oracle_sr'],
            score_summary_before['spl'],
            score_summary_before['rgs'],
            score_summary_before['rgspl'],
        )
    )

    # --------------------------------
    # Online adaptation over episodes
    # --------------------------------
    print("\n===== FEEDTTA ONLINE ADAPTATION =====")
    device = next(agent.vln_bert.parameters()).device

    # restart env stream from beginning for adaptation
    agent.env.reset_epoch(shuffle=False)

    for k in range(tta_steps):
        optimizer.zero_grad(set_to_none=True)
        agent.logs = defaultdict(list)

        agent.vln_bert.train()
        if hasattr(agent, 'critic'):
            agent.critic.eval()

        agent.feedback = 'sample'
        agent.loss = torch.zeros([], dtype=torch.float32, device=device)

        try:
            _ = agent.rollout(train_ml=None, train_rl=True, reset=True)
        except Exception as e:
            print(f"[FEEDTTA] episode {k} skipped: rollout failed: {e}")
            optimizer.zero_grad(set_to_none=True)
            continue

        rl_loss = agent.loss
        if rl_loss is None:
            print(f"[FEEDTTA] episode {k} skipped: RL loss missing")
            optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.is_tensor(rl_loss):
            rl_loss = torch.tensor(rl_loss, dtype=torch.float32, device=device)

        rl_loss_value = float(rl_loss.detach().item())
        print(f"[FEEDTTA] episode {k} RL_loss = {rl_loss_value:.6f}")

        if (not torch.isfinite(rl_loss)) or (not rl_loss.requires_grad) or (rl_loss.grad_fn is None):
            print(f"[FEEDTTA] episode {k} skipped: invalid autograd state")
            optimizer.zero_grad(set_to_none=True)
            continue

        rl_loss.backward()

        if sgr_p > 0:
            apply_sgr([agent.vln_bert], p=sgr_p, alpha=sgr_alpha)

        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.vln_bert.parameters() if p.requires_grad],
            grad_clip
        )
        optimizer.step()

    # ------------------------
    # Eval after adaptation
    # ------------------------
    print("\n===== EVAL AFTER ADAPTATION =====")
    agent.vln_bert.eval()
    if hasattr(agent, 'critic'):
        agent.critic.eval()

    agent.logs = defaultdict(list)
    agent.test(use_dropout=False, feedback='argmax', iters=None)
    preds_after = agent.get_results(detailed_output=args.detailed_output)
    score_summary_after, _ = env.eval_metrics(preds_after)

    print(
        "[AFTER] Env name: %s, action_steps: %.2f, steps: %.2f, lengths: %.2f, "
        "sr: %.2f, oracle_sr: %.2f, spl: %.2f, rgs: %.2f, rgspl: %.2f"
        % (
            target_env_name,
            score_summary_after['action_steps'],
            score_summary_after['steps'],
            score_summary_after['lengths'],
            score_summary_after['sr'],
            score_summary_after['oracle_sr'],
            score_summary_after['spl'],
            score_summary_after['rgs'],
            score_summary_after['rgspl'],
        )
    )

    print("\n===== DELTA =====")
    for key in ['action_steps', 'steps', 'lengths', 'sr', 'oracle_sr', 'spl', 'rgs', 'rgspl']:
        before_v = score_summary_before[key]
        after_v = score_summary_after[key]
        print(f"{key}: {before_v:.2f} -> {after_v:.2f}  (delta {after_v - before_v:+.2f})")

def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, rank=rank)

    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        # valid(args, train_env, val_envs, rank=rank)
        feedtta_valid(args, train_env, val_envs, rank=rank)           

if __name__ == '__main__':
    main()
