# coding: utf-8
import os
from PIL import Image
from time import sleep
import cv2
import picamera
import picamera.array
import torch
# pytorchディレクトリで "export OMP_NUM_THREADS=1 or 2 or 3" 必須(デフォルトは4)
# 並列処理コア数は "print(torch.__config__.parallel_info())" で確認
import torch.nn as nn
import torch.utils
from torchvision import transforms

CKPT_NET = 'trained_net.ckpt'  # 学習済みパラメータファイル
# OBJ_NAMES = ['Phone', 'Wallet', 'Watch']  # 各クラスの表示名
OBJ_NAMES = ['Curtain', 'Phone']  # 各クラスの表示名
MIN_LEN = 50
GRAY_THR = 20
CONTOUR_COUNT_MAX = 3  # バッチサイズ(一度に検出する物体の数)の上限
# CONTOUR_COUNT_MAX = 2  # バッチサイズ(一度に検出する物体の数)の上限
SHOW_COLOR = (255, 191, 0)  # 枠の色(B,G,R)

# NUM_CLASSES = 3
NUM_CLASSES = 2
PIXEL_LEN = 112  # Resize後のサイズ(1辺)
CHANNELS = 1  # 色のチャンネル数(BGR:3, グレースケール:1)


# 画像データ変換定義
# Resizeと, classifierの最初のLinear入力が関連
data_transforms = transforms.Compose([
    transforms.Resize((PIXEL_LEN, PIXEL_LEN)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class NeuralNet(nn.Module):
    """ネットワーク定義. 学習に用いたものと同一である必要"""
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


def detect_obj(back, target):
    """
    OpenCVの背景差分処理で, 検出された物体のタプルを作成
    引数:
        back: 入力背景画像
            カラー画像
        target: 背景差分対象の画像
            カラー画像. 複数の物体を切り抜き, カラー画像タプルにまとめる
    """
    print('Detecting objects ...')
    # 2値化
    b_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # 差分を計算
    diff = cv2.absdiff(t_gray, b_gray)

    # 閾値に従ってコンター, マスクを作成, 物体を抽出
    # findContoursのインデックスは, cv2.__version__ == 4.2.0->[0], 3.4.7->[1]
    mask = cv2.threshold(diff, GRAY_THR, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('mask', mask)
    contour = cv2.findContours(mask,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[0]

    # 一定の縦横幅以上で検出された変化領域の座標, サイズバッチを作成
    pt_list = list(filter(
        lambda x: x[2] > MIN_LEN and x[3] > MIN_LEN,
        [cv2.boundingRect(pt) for pt in contour]
    ))[:CONTOUR_COUNT_MAX]

    # 位置情報に従ってフレーム切り抜き, PIL画像のタプルに変換して返す
    obj_imgaes = tuple(map(
        lambda x: Image.fromarray(target[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]),
        pt_list
    ))
    return (obj_imgaes, pt_list)


def batch_maker(tuple_images, transform):
    """
    PIL形式の画像のタプルをtransformし, ネットワークで処理可能なtensorバッチに変換
    引数:
        tuple_images: PIL画像タプル
        transform: torchvision画像変換定義
    """
    return torch.cat([transform(img) for img
                      in tuple_images]).view(-1, CHANNELS, PIXEL_LEN, PIXEL_LEN)


def judge_what(img, probs_list, pos_list):
    """
    各クラスに属する確率から物体を決定し, その位置に枠と名前を表示, クラスのインデックスを返す
    引数:
        probs_list: 確率の二次配列. バッチ形式
        pos_list: 位置の二次配列. バッチ形式
    """
    print('Judging objects ...')
    # 最も高い確率とそのインデックスのリストに変換
    # ip_list = list(map(lambda x: max(enumerate(x), key = lambda y:y[1]),
    #                    F.softmax(probs_list, dim=-1)))  # <- 4/30修正
    ip_list = list(map(lambda x: max(enumerate(x), key = lambda y:y[1]),
                       torch.nn.functional.softmax(probs_list, dim=-1)))  # <- 4/30修正

    # インデックスを物体名に変換, 物体の位置に物体名と確信度を書き込み表示
    for (idx, prob), pos in zip(ip_list, pos_list):
        cv2.rectangle(img, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), SHOW_COLOR, 2)
        cv2.putText(img, '%s:%.1f%%'%(OBJ_NAMES[idx], prob*100), (pos[0]+5, pos[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, SHOW_COLOR, thickness=2)
    return ip_list


def realtime_classify():
    """学習済みモデル読み込み->テストデータ読み込み->分類->結果を画像に重ねて表示"""
    # デバイス設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ネットワーク設定
    net = NeuralNet(NUM_CLASSES).to(device)

    # 訓練済みデータ取得
    if os.path.isfile(CKPT_NET):
        checkpoint = torch.load(CKPT_NET)
        net.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError('No trained network file: {}'.format(CKPT_NET))

    # 評価モード
    net.eval()
    # picamera起動
    with picamera.PiCamera() as camera:
        camera.resolution = (480, 480)
        # ストリーミング開始
        with picamera.array.PiRGBArray(camera) as stream:
            print('Setting background ...')
            sleep(2)

            camera.exposure_mode = 'off'  # ホワイトバランス固定
            camera.capture(stream, 'bgr', use_video_port=True)
            # 背景に設定
            img_back = stream.array

            stream.seek(0)
            stream.truncate()

            print('Start!')
            with torch.no_grad():
                while True:
                    camera.capture(stream, 'bgr', use_video_port=True)
                    # これからの入力画像に対して背景差分
                    img_target = stream.array
                    # 物体とその位置を検出
                    obj_imgs, positions = detect_obj(img_back, img_target)
                    if obj_imgs:
                        # 検出物体をネットワークの入力形式に変換
                        obj_batch = batch_maker(obj_imgs, data_transforms)
                        # 分類
                        outputs = net(obj_batch)
                        # 判定
                        result = judge_what(img_target, outputs, positions)
                        print('  Result:', result)

                    # 表示                    
                    cv2.imshow('detection', img_target)

                    if cv2.waitKey(200) == ord('q'):
                        cv2.destroyAllWindows()
                        return

                    stream.seek(0)
                    stream.truncate()


if __name__ == "__main__":
    try:
        realtime_classify()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
