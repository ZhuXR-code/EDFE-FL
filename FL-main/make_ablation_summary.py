# -*- coding: utf-8 -*-
"""
消融实验结果汇总（思路移植自参考项目 make_ablation_summary.py / FL攻击实验汇总 md）

扫描 ablation_logs/*.log，提取终版脚本打印的 [ABLATION][RESULT] 摘要行，
解析成表格并输出 CSV + Markdown，便于写论文实验章节。

用法：
  python make_ablation_summary.py                    # 汇总所有 log
  python make_ablation_summary.py --out summary.xlsx # 额外输出 Excel（需 pandas+openpyxl）
"""

import argparse
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, 'ablation_logs')

# [ABLATION][RESULT] rule=... attack=... ratio=... alpha=... final_train_acc=... test_acc=... tag=...
RESULT_RE = re.compile(
    r'\[ABLATION\]\[RESULT\]\s+rule=(\S+)\s+attack=(\S+)\s+ratio=(\S+)\s+'
    r'alpha=(\S+)\s+final_train_acc=(\S+)\s+test_acc=(\S+)\s+tag=(\S+)'
)


def collect(log_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, '*.log'))):
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = RESULT_RE.search(line)
                if not m:
                    continue
                rule, attack, ratio, alpha, train_acc, test_acc, tag = m.groups()
                rows.append({
                    'rule': rule, 'attack': attack, 'ratio': ratio, 'alpha': alpha,
                    'final_train_acc': train_acc, 'test_acc': test_acc,
                    'tag': tag, 'log': os.path.basename(path),
                })
    return rows


def write_csv(rows, out_path):
    if not rows:
        print('没有可汇总的结果行'); return
    keys = list(rows[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'CSV 已写入: {out_path}（{len(rows)} 行）')


def write_markdown(rows, out_path):
    if not rows:
        return
    keys = ['rule', 'attack', 'ratio', 'alpha', 'test_acc', 'final_train_acc', 'tag']
    lines = ['# EDFE-FE-FL 消融实验结果汇总', '',
             '> 由 `make_ablation_summary.py` 从 `ablation_logs/*.log` 自动提取。',
             '', '| ' + ' | '.join(keys) + ' |',
             '|' + '|'.join(['---'] * len(keys)) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(r[k]) for k in keys) + ' |')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Markdown 已写入: {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log-dir', default=LOG_DIR)
    p.add_argument('--csv', default=os.path.join(HERE, 'ablation_summary.csv'))
    p.add_argument('--md', default=os.path.join(HERE, '消融实验汇总.md'))
    p.add_argument('--xlsx', default=None)
    args = p.parse_args()

    rows = collect(args.log_dir)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    if args.xlsx:
        try:
            import pandas as pd
            pd.DataFrame(rows).to_excel(args.xlsx, index=False)
            print(f'Excel 已写入: {args.xlsx}')
        except ImportError:
            print('未安装 openpyxl/pandas，跳过 xlsx 输出')


if __name__ == '__main__':
    main()
