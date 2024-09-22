# coding: utf-8
import cv2
from datetime import datetime
import picamera
import picamera.array

MIN_LEN = 50  # 物体検出枠の1辺の最小長さ
GRAY_THR = 20  # 濃度変化の閾値
CUT_MODE = True  # True:検出物体を切り取って保存, False:画像全体をそのまま保存
PATH="/home/pi/open-cv/images/"


"""
取得画像中の物体検出箇所全てを四角枠で囲む
引数:
    img: カメラ画像
    contour: コンター
    minlen: 検出の大きさの閾値（これより枠の1辺が短い箇所は除く）
"""
def imshow_rect(img, contour, minlen=0):
    for pt in contour:
        x, y, w, h = cv2.boundingRect(pt)
        if w < minlen and h < minlen: continue
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Preview', img)


"""
取得画像中の物体検出箇所を全て切り抜き保存
引数:
    同上
"""
def save_cutimg(img, contour, minlen=0):
    # 日時を取得しファイル名に使用
    dt = datetime.now()
    f_name = '{}.jpg'.format(dt.strftime('%y%m%d%H%M%S'))
    imgs_cut = []
    for pt in contour:
        x, y, w, h = cv2.boundingRect(pt)
        if w < minlen and h < minlen: continue
        imgs_cut.append(img[y:y+h, x:x+w])

    # 物体を切り抜いて保存
    if not imgs_cut: return -1
    if len(imgs_cut) > 1:
        for i in range(len(imgs_cut)):
            cv2.imwrite(PATH+f_name[:-4]+'_'+str(i+1)+f_name[-4:], imgs_cut[i])
    else:
        cv2.imwrite(PATH+f_name, imgs_cut[0])
    return len(imgs_cut)


"""
取得画像をそのまま保存
引数:
    同上
"""
def save_img(img):
    dt = datetime.now()
    fname = '{}.jpg'.format(dt.strftime('%y%m%d%H%M%S'))
    cv2.imwrite(PATH+fname, img)


"""
背景撮影->物体撮影, 保存
キー入力: 
    "p": 写真を撮る
    "q": やめる
    "r": 画面を回転（背景撮影時）
    "i": 初めからやり直す（物体撮影時）
"""
def take_photo():
    cnt = 0
    # picamera起動
    with picamera.PiCamera() as camera:
        camera.resolution = (480, 480)  # 解像度
        camera.rotation = 0  # カメラの回転角(度)
        # ストリーミング開始
        with picamera.array.PiRGBArray(camera) as stream:
            print('Set background ... ', end='', flush=True)
            # 初めに背景を撮影
            while True:
                # ストリーミング画像を取得、表示
                camera.capture(stream, 'bgr', use_video_port=True)
                cv2.imshow('Preview', stream.array)

                wkey = cv2.waitKey(5) & 0xFF  # キー入力受付

                stream.seek(0)  # 新しくcaptureするための呪文x2
                stream.truncate()

                if wkey == ord('q'):
                    cv2.destroyAllWindows()
                    return print()
                elif wkey == ord('r'):
                    camera.rotation += 90
                elif wkey == ord('p'):
                    camera.exposure_mode = 'off'  # ホワイトバランス固定
                    save_img(stream.array)
                    # グレースケール化して背景画像に設定
                    back_gray = cv2.cvtColor(stream.array, 
                                             cv2.COLOR_BGR2GRAY)
                    print('done')
                    break

            # 背景を設定し終えたら, カメラを動かさないように対象物撮影
            print('Take photos!')
            while True:
                camera.capture(stream, 'bgr', use_video_port=True)
                # 現在のフレームをグレースケール化
                stream_gray = cv2.cvtColor(stream.array, 
                                           cv2.COLOR_BGR2GRAY)

                # 差分の絶対値を計算し二値化, マスク作成
                diff = cv2.absdiff(stream_gray, back_gray)
                mask = cv2.threshold(diff, GRAY_THR, 255, 
                                     cv2.THRESH_BINARY)[1]
                cv2.imshow('mask', mask)

                # 物体検出のためのコンター, マスク作成
                # contour = cv2.findContours(mask,
                #                            cv2.RETR_EXTERNAL,
                #                            cv2.CHAIN_APPROX_SIMPLE)[1]
                contour = cv2.findContours(mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)[0]

                # 検出された物体全てを四角で囲み表示
                stream_arr = stream.array.copy()
                imshow_rect(stream_arr, contour, MIN_LEN)

                wkey = cv2.waitKey(5) & 0xFF

                stream.seek(0)
                stream.truncate()

                if wkey == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif wkey == ord('i'):
                    break
                elif wkey == ord('p'):
                    if CUT_MODE:
                        num = save_cutimg(stream.array, contour, MIN_LEN)
                        if num > 0:
                            cnt += num
                            print('  Captured: {} (sum: {})'.format(num, cnt))
                    else:
                        save_img(stream.array)
                        cnt += 1
                        print('  Captured: 1 (sum: {})'.format(cnt))

    print('Initialized')
    take_photo()


if __name__ == '__main__':
    take_photo()
