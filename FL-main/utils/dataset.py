import random

import numpy as np
from torchvision import datasets,transforms


def sample_dirichlet(dataset, num_users, alpha=1.0, num_classes=10):
    """Dirichlet(alpha) 非 IID 划分（移植自参考项目 sampling.py:sample_dirichlet）。

    给每个类别按 Dirichlet(alpha) 采样各客户端的比例，再按比例切分该类的样本索引。
    alpha 越小，各客户端的类别分布越尖锐（非 IID 程度越高）。

    返回: {user_id: np.ndarray(样本索引)}，格式与本项目 user_samples 一致。
    """
    classes = {c: [] for c in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        if hasattr(label, 'item'):
            label = label.item()
        classes[int(label)].append(idx)

    peers = {i: np.array([], dtype=int) for i in range(num_users)}
    for c in range(num_classes):
        idxs = classes[c]
        random.shuffle(idxs)
        probs = np.random.dirichlet(np.array([alpha] * num_users))
        num_each = np.array([int(round(p * len(idxs))) for p in probs])
        # 处理取整导致的余量/超额：多退少补，保证该类样本全部分配且不重复
        diff = len(idxs) - num_each.sum()
        if diff > 0:
            order = np.argsort(-probs)
            for k in range(diff):
                num_each[order[k % num_users]] += 1
        elif diff < 0:
            order = np.argsort(probs)
            for k in range(-diff):
                i = order[k % num_users]
                if num_each[i] > 0:
                    num_each[i] -= 1
        start = 0
        for user in range(num_users):
            take = num_each[user]
            if take > 0:
                peers[user] = np.concatenate((peers[user], np.array(idxs[start:start + take], dtype=int)))
                start += take
    return peers


def mnist_iid(dataset,num_users):
    num_samples = int(len(dataset) / num_users)
    index_of_samples = [i for i in range(len(dataset))]
    user_samples = {}
    for k in range(num_users):
        user_samples[k] = set(np.random.choice(index_of_samples,num_samples,replace=False))
        index_of_samples = list(set(index_of_samples) - user_samples[k])
    return user_samples


def mnist_noniid(dataset,num_users):
    num_shards,num_samples = 300,200
    index_of_shards = [i for i in range(num_shards)]
    user_samples = {k : np.array([]) for k in range(num_users)}
    index_of_samples = np.arange(num_shards * num_samples)
    labels = dataset.train_labels.numpy()
    samples_labels = np.vstack((index_of_samples,labels))
    samples_labels = samples_labels[:,samples_labels[1,:].argsort()]
    index_of_samples = samples_labels[0,:]

    for k in range(num_users):
        random_2_shards = set(np.random.choice(index_of_shards, int(num_shards / num_users), replace=False))
        index_of_shards = list(set(index_of_shards) - random_2_shards)
        for shard in random_2_shards:
            user_samples[k] = np.concatenate(
                (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

    return user_samples


def mnist_noniid_unequal(dataset,num_users):
    num_shards,num_samples = 1200,50
    index_of_shards = [i for i in range(num_shards)]
    user_samples = {k : np.array([]) for k in range(num_users)}
    index_of_samples = np.arange(num_shards * num_samples)
    labels = dataset.train_labels.numpy()
    samples_labels = np.vstack((index_of_samples, labels))
    samples_labels = samples_labels[:, samples_labels[1, :].argsort()]
    index_of_samples = samples_labels[0, :]
    min_shards = 1
    max_shards = 30
    random_shards_size = np.random.randint(min_shards,max_shards + 1,size=num_users)
    random_shards_size = np.around(random_shards_size / sum(random_shards_size)
                                   * num_shards)
    random_shards_size = random_shards_size.astype(int)

    if sum(random_shards_size) > num_shards:
        for k in range(num_users):
            random_1_shard = set(np.random.choice(index_of_shards,1,replace=False))
            index_of_shards = list(set(index_of_shards) - random_1_shard)
            for shard in random_1_shard:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

        random_shards_size = random_shards_size - 1

        for k in range(num_users):
            if len(index_of_shards) == 0:
                continue
            shards_size = random_shards_size[k]
            if shards_size > len(index_of_shards):
                shards_size = len(index_of_shards)
            random_shards = set(np.random.choice(index_of_shards,shards_size,replace=False))
            index_of_shards = list(set(index_of_shards) - random_shards)
            for shard in random_shards:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

    else:
        for k in range(num_users):
            shards_size = random_shards_size[k]
            random_shards = set(np.random.choice(index_of_shards,shards_size,replace=False))
            index_of_shards = list(set(index_of_shards) - random_shards)
            for shard in random_shards:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

        if len(index_of_shards) > 0:
            shards_size = len(index_of_shards)
            k = min(user_samples,key=lambda x: len(user_samples.get(x)))
            random_shards = set(np.random.choice(index_of_shards,shards_size,replace=False))
            index_of_shards = list(set(index_of_shards) - random_shards)
            for shard in random_shards:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

    return user_samples


def cifar_iid(dataset,num_users):
    num_samples = int(len(dataset) / num_users)
    index_of_samples = [i for i in range(len(dataset))]
    user_samples = {}

    for k in range(num_users):
        user_samples[k] = set(np.random.choice(index_of_samples, num_samples, replace=False))
        index_of_samples = list(set(index_of_samples) - user_samples[k])

    return user_samples


def cifar_noniid(dataset,num_users):
    num_shards, num_samples = 200, 250
    index_of_shards = [i for i in range(num_shards)]
    user_samples = {k: np.array([]) for k in range(num_users)}
    index_of_samples = np.arange(num_shards * num_samples)
    labels = np.array(dataset.targets)
    samples_labels = np.vstack((index_of_samples, labels))
    samples_labels = samples_labels[:, samples_labels[1, :].argsort()]
    index_of_samples = samples_labels[0, :]

    for k in range(num_users):
        random_2_shards = set(np.random.choice(index_of_shards, 2, replace=False))
        index_of_shards = list(set(index_of_shards) - random_2_shards)
        for shard in random_2_shards:
            user_samples[k] = np.concatenate(
                (user_samples[k], index_of_samples[shard * num_samples: (shard + 1) * num_samples]), axis=0)

    return user_samples


def get_dataset(args):
  
    train_dataset,test_dataset,user_samples = None,None,None

    if args.dataset == 'cifar':
        data_dir = '../data/cifar'

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        train_dataset = datasets.CIFAR10(data_dir,train=True,download=True,
                                         transform=transform)
        test_dataset = datasets.CIFAR10(data_dir,train=False,download=True,
                                        transform=transform)

        if getattr(args, 'alpha', None) is not None and not args.iid:
            # Dirichlet 非 IID 划分（消融实验用）
            user_samples = sample_dirichlet(train_dataset, args.num_users,
                                            alpha=args.alpha, num_classes=args.num_classes)
        elif args.iid:
            user_samples = cifar_iid(train_dataset,args.num_users)
        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                user_samples = cifar_noniid(train_dataset,args.num_users)


    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist'
        else:
            data_dir = '../data/fmnist'

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081))])

        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir,train=True,download=True,
                                           transform=transform)
            test_dataset = datasets.MNIST(data_dir,train=False,download=True,
                                          transform=transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                          transform=transform)


        if getattr(args, 'alpha', None) is not None and not args.iid:
            # Dirichlet 非 IID 划分（消融实验用）
            user_samples = sample_dirichlet(train_dataset, args.num_users,
                                            alpha=args.alpha, num_classes=args.num_classes)
        elif args.iid:
            user_samples = mnist_iid(train_dataset,args.num_users)
        else:
            if args.unequal:
                user_samples = mnist_noniid_unequal(train_dataset,args.num_users)
            else:
                user_samples = mnist_noniid(train_dataset,args.num_users)

    return train_dataset,test_dataset,user_samples
