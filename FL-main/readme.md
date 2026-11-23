
修改参数和配置文件在utils下修改
外围是运行文件

外面的分别是不同的方式
dp、edfe、nddfe、pyfhel

pyfhel 注意安装的版本不同，参数用的不同

=====================================================================
鲁棒聚合 / 攻击 / 消融实验（移植自参考项目 FL+SV+JSD-zxr-v3.0）
=====================================================================
参考项目在明文联邦学习上实现了 Shapley Value + JSD-Krum 鲁棒聚合与多种攻击/消融实验。
本次把它的算法与实验能力移植到本加密项目：服务器端解密后可见每个客户端的明文更新，
防御规则作用于这份明文更新，加密求和主流程保持不变。

新增文件：
  utils/robust_aggregation.py  鲁棒聚合算法（JSD-Krum B3版 / Krum / JSD-only /
                               median / trimmed-mean / Shapley / FoolsGold / 加权平均）
  utils/attacks.py             攻击注入（label_flipping / gaussian / sign_flip 及数据集标签翻转）
  run_ablation.py              rule × ratio 消融实验编排（.claims 防重 + 配置签名）
  make_ablation_summary.py     从日志提取结果汇总为 CSV / Markdown / Excel

改造文件：
  utils/options.py             新增 --rule --attack-type --malicious-behavior-rate --f
                               --alpha --source-class --target-class --sign-flip-mode
                               --gauss-std --gauss-relative --gauss-rel-std
                               --sv-permutations --tag
  utils/dataset.py             新增 Dirichlet(alpha) 非 IID 划分（--alpha 且 --iid 0 时启用）
  EDFE_FL_main_自适应缩放.py / EDFE_FL_main_固定缩放.py
                              客户端训练接入攻击；解密后按 --rule 做防御聚合；
                              结果与曲线文件名带 rule/attack/ratio/tag 标识
                              两脚本仅量化参数取法不同：前者按各客户端权重自适应计算
                              scale/offset 后取 max 统一（量化误差~1e-15，精度优先）；
                              后者全局写死 scale=100/offset=10（误差~5e-3，简洁基线）

运行示例（在这些脚本所在目录执行，用项目实际环境的 python）：
  # 基线 fedavg
  python EDFE_FL_main_自适应缩放.py --rule fedavg --epochs 50

  # JSD-Krum 防御 + 30% 标签翻转攻击（f 自动 = ceil(0.3*num_users)）
  python EDFE_FL_main_自适应缩放.py --rule jsd-krum --attack-type label_flipping \
      --malicious-behavior-rate 0.3 --epochs 50

  # sign_flip delta 档
  python EDFE_FL_main_自适应缩放.py --rule jsd-krum --attack-type sign_flip \
      --sign-flip-mode delta --malicious-behavior-rate 0.3 --epochs 50

  # 批量消融（3 个防御 × 3 个比例）
  python run_ablation.py --rules fedavg krum jsd-krum --ratios 0.1 0.3 0.5 \
      --attack-type label_flipping --epochs 50 --num_users 20 --local_ep 2

  # 结果汇总
  python make_ablation_summary.py --xlsx 消融实验汇总.xlsx

关键算法口径（务必保持，来自参考项目 B3 方案，改动会导致消融数据不可比）：
  JSD-Krum 的 k = n - f - 2（不做 max(1)）；topk 含自身距离（对角线置 0）；
  eps = np.finfo(float).eps。已用 smoke 脚本验证与参考项目保留集完全一致。

说明：run_ablation.py 调用终版脚本需要项目真实环境（WSL + Charm/PBC + torch）。
Windows 侧仅做语法/算法自检，不含加密库运行时。