# -*- coding: utf-8 -*-
import picamera
import picamera.array
import cv2 as cv
import time

# カメラ初期化
with picamera.PiCamera() as camera:
    # カメラの画像をリアルタイムで取得するための処理
    with picamera.array.PiRGBArray(camera) as stream:
        # 解像度の設定
        # camera.resolution = (512, 384)
        camera.resolution = (480, 480)
        # camera.resolution = (3280, 2464)
        # # 撮影の準備
        # camera.start_preview()
        # # 準備している間、少し待機する
        # time.sleep(2)

        while True:
            # カメラから映像を取得する（OpenCVへ渡すために、各ピクセルの色の並びをBGRの順番にする）
            camera.capture(stream, 'bgr', use_video_port=True)
            # 顔検出の処理効率化のために、写真の情報量を落とす（モノクロにする）
            grayimg = cv.cvtColor(stream.array, cv.COLOR_BGR2GRAY)
            # 顔検出のための学習元データを読み込む
            face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            # 顔検出を行う
            facerect = face_cascade.detectMultiScale(grayimg, scaleFactor=1.2, minNeighbors=2, minSize=(100, 100))
            # 顔が検出された場合
            if len(facerect) > 0:
                # そのときの画像を保存する
                cv.imwrite('my_pic2.jpg', stream.array)
                break
            # カメラから取得した映像を表示する
            cv.imshow('camera', stream.array)
            # カメラから読み込んだ映像を破棄する
            stream.seek(0)
            stream.truncate()
            
            # 何かキーが押されたかどうかを検出する（検出のため、1ミリ秒待つ）
            if cv.waitKey(1) > 0:
                break

        # 表示したウィンドウを閉じる
        cv.destroyAllWindows()
