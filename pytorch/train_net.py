# coding: utf-8
import os
import re
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DATA_DIR = 'image_data'  # 画像フォルダ名
CKPT_PROCESS = 'train_process.ckpt'  # 学習経過保存ファイル名
CKPT_NET = 'trained_net.ckpt'  # 学習済みパラメータファイル名
NUM_CLASSES = 2  # クラス数
# NUM_CLASSES = 3  # クラス数
NUM_EPOCHS = 100  # 学習回数

# よく変更するハイパーパラメータ
LEARNING_RATE = 0.001  # 学習率
MOMENTUM = 0.5  # 慣性

checkpoint = {}  # 途中経過保存用変数


# 画像データ変換定義（かさ増し）
# Resizeのサイズと, classifierの最初のLinear入力サイズが関連
train_transforms = transforms.Compose([
    transforms.Resize((112, 112)),  # リサイズ
    transforms.RandomRotation(30),  # ランダムに回転
    transforms.Grayscale(),  # 2値化
    transforms.ToTensor(),  # テンソル化
    transforms.Normalize(mean=[0.5], std=[0.5])  # 正規化（数字はテキトー）
])

val_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# データセット作成
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'train'),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'val'),
    transform=val_transforms
)

# ミニバッチ取得
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=10,  # 学習時のバッチサイズ
    shuffle=True  # 訓練データをシャッフル
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=10,
    shuffle=True
)


class NeuralNet(nn.Module):
    """ネットワーク定義. nn.Module継承"""
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    """訓練途中データ読み込み->学習(->訓練途中データの保存)->結果の図示"""
    global checkpoint
    print('[Settings]')
    # デバイスの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'device:{device}')

    # ネットワーク, 評価関数, 最適化関数設定
    net = NeuralNet(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()  # 評価関数
    optimizer = optim.SGD(  # 最適化アルゴリズム
        net.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=5e-4
    )

    # 設定の表示
    # print('  Device               :', device)
    # print('  Dataset Class-Index  :', train_dataset.class_to_idx)
    # print('  Network Model        :', re.findall('(.*)\(', str(net))[0])
    # print('  Criterion            :', re.findall('(.*)\(', str(criterion))[0])
    # print('  Optimizer            :', re.findall('(.*)\(', str(optimizer))[0])
    # print('    -Learning Rate     :', LEARNING_RATE)
    # print('    -Momentum          :', MOMENTUM)

    t_loss_list = []
    t_acc_list = []
    v_loss_list = []
    v_acc_list = []
    epoch_pre = -1

    # 訓練（途中）データ取得
    if os.path.isfile(CKPT_PROCESS):
        checkpoint = torch.load(CKPT_PROCESS)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        t_loss_list = checkpoint['t_loss_list']
        t_acc_list = checkpoint['t_acc_list']
        v_loss_list = checkpoint['v_loss_list']
        v_acc_list = checkpoint['v_acc_list']
        epoch_pre = checkpoint['epoch']
        print("Progress until last time = {}/{} epochs"\
              .format(epoch_pre+1, NUM_EPOCHS))

    print('[Main process]')
    for epoch in range(epoch_pre+1, NUM_EPOCHS):
        t_loss, t_acc, v_loss, v_acc = 0, 0, 0, 0

        # 学習 ---------------------------------------------------------
        net.train()  # 学習モード
        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            t_loss += loss.item()
            t_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()
        avg_t_loss = t_loss / len(train_loader.dataset)
        avg_t_acc = t_acc / len(train_loader.dataset)

        # 評価 ---------------------------------------------------------
        net.eval()  # 評価モード
        with torch.no_grad():  # 勾配の更新を停止
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item()
                v_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_v_loss = v_loss / len(val_loader.dataset)
        avg_v_acc = v_acc / len(val_loader.dataset)
        # --------------------------------------------------------------
        print('\rEpoch [{}/{}] | Train [oss:{:.3f}, acc:{:.3f}] | Val [loss:{:.3f}, acc:{:.3f}]'\
              .format(epoch+1, NUM_EPOCHS, avg_t_loss, avg_t_acc, avg_v_loss, avg_v_acc), end='')

        # 損失, 精度記録
        t_loss_list.append(avg_t_loss)
        t_acc_list.append(avg_t_acc)
        v_loss_list.append(avg_v_loss)
        v_acc_list.append(avg_v_acc)

        # 途中経過保存用処理
        checkpoint['net'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['t_loss_list'] = t_loss_list
        checkpoint['t_acc_list'] = t_acc_list
        checkpoint['v_loss_list'] = v_loss_list
        checkpoint['v_acc_list'] = v_acc_list
        checkpoint['epoch'] = epoch

    graph()
    save_process()
    save_net()


def save_process():
    """途中経過を保存"""
    global checkpoint
    if not checkpoint: return
    torch.save(checkpoint, CKPT_PROCESS)


def save_net():
    """ネットワーク情報のみ保存"""
    global checkpoint
    if not checkpoint: return
    torch.save(checkpoint['net'], CKPT_NET)


def graph():
    """損失, 精度のグラフ化"""
    global checkpoint
    if not checkpoint: return
    t_loss_list = checkpoint['t_loss_list']
    t_acc_list = checkpoint['t_acc_list']
    v_loss_list = checkpoint['v_loss_list']
    v_acc_list = checkpoint['v_acc_list']

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(t_loss_list)), t_loss_list,
             color='blue', linestyle='-', label='t_loss')
    plt.plot(range(len(v_loss_list)), v_loss_list,
             color='green', linestyle='--', label='v_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(t_acc_list)), t_acc_list,
             color='blue', linestyle='-', label='t_acc')
    plt.plot(range(len(v_acc_list)), v_acc_list,
             color='green', linestyle='--', label='v_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('Training and validation accuracy')
    plt.grid()
    plt.savefig("graph.png")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        graph()
        save_process()
