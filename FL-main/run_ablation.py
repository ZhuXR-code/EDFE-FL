# -*- coding: utf-8 -*-
"""
消融实验编排脚本（思路移植自参考项目 FL+SV+JSD-zxr-v3.0/run_stage.py）

对「防御规则 rule × 恶意比例 ratio」做双层循环，调用加密主脚本跑实验，
并把每组的配置签名写入 .claims 目录防止重复运行（并发安全：文件已存在即跳过）。

用法示例：
  python run_ablation.py --script EDFE_FL_main_自适应缩放.py \
      --rules fedavg krum jsd-krum jsd-only --ratios 0.1 0.3 0.5 \
      --attack-type label_flipping --epochs 50 --num-users 20 --local-ep 2

  # sign_flip delta 档
  python run_ablation.py --rules fedavg jsd-krum --ratios 0.3 \
      --attack-type sign_flip --sign-flip-mode delta

  # alpha 敏感性
  python run_ablation.py --rules fedavg jsd-krum --ratios 0.3 \
      --attack-type label_flipping --alpha 0.1 --tag alpha01

每组会生成一个 log 文件：ablation_logs/<rule>_<attack>_<ratio>_<tag>.log
结果摘要行（[ABLATION][RESULT]）也会被汇总脚本 make_ablation_summary.py 读取。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CLAIMS_DIR = os.path.join(HERE, '.claims_ablation')
LOG_DIR = os.path.join(HERE, 'ablation_logs')


def config_signature(args, rule, ratio):
    """配置签名：这些项变了就是不同实验，不能与旧结果混用。"""
    keys = ['script', 'dataset', 'num_users', 'epochs', 'local_ep', 'frac',
            'attack_type', 'source_class', 'target_class', 'sign_flip_mode',
            'alpha', 'gauss_std', 'gauss_relative', 'gauss_rel_std', 'seed']
    d = {'script': args.script, 'rule': rule, 'ratio': ratio}
    for k in keys:
        d[k] = getattr(args, k, None)
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.md5(blob).hexdigest()[:12]


def run_one(args, rule, ratio):
    sig = config_signature(args, rule, ratio)
    claim_file = os.path.join(CLAIMS_DIR, f'{sig}.claim')
    tag = args.tag or 'base'
    log_file = os.path.join(LOG_DIR, f'{rule}_{args.attack_type}_{ratio}_{tag}.log')

    if os.path.exists(claim_file) and not args.force:
        print(f'[SKIP] rule={rule} ratio={ratio} sig={sig}（已认领，--force 可强制重跑）')
        return

    cmd = [
        sys.executable, args.script,
        '--rule', rule,
        '--malicious-behavior-rate', str(ratio),
        '--attack-type', args.attack_type,
        '--epochs', str(args.epochs),
        '--num_users', str(args.num_users),
        '--local_ep', str(args.local_ep),
        '--frac', str(args.frac),
        '--seed', str(args.seed),
        '--source-class', str(args.source_class),
        '--target-class', str(args.target_class),
        '--sign-flip-mode', args.sign_flip_mode,
        '--gauss-std', str(args.gauss_std),
        '--gauss-rel-std', str(args.gauss_rel_std),
        '--tag', f'{tag}_{sig}',
    ]
    if args.alpha is not None:
        cmd += ['--alpha', str(args.alpha), '--iid', '0']
    if args.gauss_relative:
        cmd += ['--gauss-relative', '1']

    # 写认领标记（含完整配置，便于自证）
    with open(claim_file, 'w', encoding='utf-8') as f:
        json.dump({'sig': sig, 'rule': rule, 'ratio': ratio, 'cmd': cmd,
                   'time': datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

    print(f'[RUN ] rule={rule} ratio={ratio} sig={sig}')
    print('       ' + ' '.join(cmd))
    with open(log_file, 'w', encoding='utf-8') as lf:
        lf.write(' '.join(cmd) + '\n\n')
        lf.flush()
        proc = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
    print(f'[DONE] rule={rule} ratio={ratio} exit={proc.returncode} log={log_file}')


def main():
    p = argparse.ArgumentParser(description='EDFE-FE-FL 消融实验编排')
    p.add_argument('--script', default='EDFE_FL_main_自适应缩放.py',
                   help='要运行的加密主脚本名（自适应缩放 / 固定缩放）')
    p.add_argument('--rules', nargs='+',
                   default=['fedavg', 'krum', 'jsd-krum', 'jsd-only'])
    p.add_argument('--ratios', nargs='+', type=float, default=[0.1, 0.3, 0.5])
    p.add_argument('--attack-type', default='label_flipping',
                   choices=['no_attack', 'label_flipping', 'gaussian', 'sign_flip'])
    p.add_argument('--dataset', default='mnist')
    p.add_argument('--num-users', type=int, default=50)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--local-ep', type=int, default=10)
    p.add_argument('--frac', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--source-class', type=int, default=6)
    p.add_argument('--target-class', type=int, default=9)
    p.add_argument('--sign-flip-mode', default='weight', choices=['weight', 'delta'])
    p.add_argument('--gauss-std', type=float, default=0.5)
    p.add_argument('--gauss-relative', type=int, default=0)
    p.add_argument('--gauss-rel-std', type=float, default=0.0)
    p.add_argument('--alpha', type=float, default=None)
    p.add_argument('--tag', default=None)
    p.add_argument('--force', action='store_true', help='忽略已认领标记强制重跑')
    args = p.parse_args()

    os.makedirs(CLAIMS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f'消融实验：{len(args.rules)} rules × {len(args.ratios)} ratios = '
          f'{len(args.rules) * len(args.ratios)} 组')
    print(f'attack={args.attack_type}  alpha={args.alpha}  script={args.script}\n')
    for rule in args.rules:
        for ratio in args.ratios:
            run_one(args, rule, ratio)


if __name__ == '__main__':
    main()
