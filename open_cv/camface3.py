# -*- coding: utf-8 -*-
import picamera
import picamera.array
import cv2 as cv

# カメラ初期化
with picamera.PiCamera() as camera:
    # カメラの画像をリアルタイムで取得するための処理
    with picamera.array.PiRGBArray(camera) as stream:
        # 解像度の設定
        # camera.resolution = (512, 384)
        camera.resolution = (480, 480)

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
                # 検出した場所すべてに赤色で枠を描画する
                for rect in facerect:
                    cv.rectangle(stream.array, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=3)

            # 結果の画像を表示する
            cv.imshow('camera', stream.array)

            # カメラから読み込んだ映像を破棄する
            stream.seek(0)
            stream.truncate()
            
            # 何かキーが押されたかどうかを検出する（検出のため、1ミリ秒待つ）
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # 表示したウィンドウを閉じる
        cv.destroyAllWindows()
