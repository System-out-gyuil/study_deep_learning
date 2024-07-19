# 🚩 Deep learning Pretrained model project

## 해파리 이미지 분류

> https://www.kaggle.com/datasets/anshtanwar/jellyfish-types

### 목차

### 1. 데이터(이미지) 탐색 및 분리

- 총 여섯개의 타겟을 가진 약 900개의 이미지로 이루어진 데이터세트입니다.

- 이미지는 아래와 같이 바닷속, 해변가 등에서 포착된 해파리들의 이미지입니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/122436fb-6883-42f3-b812-a96d60a708c0">

<br/>

- 해당 데이터세트는 train, test, validation 데이터가 각각 폴더로 나누어져있었습니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/477473bf-58b2-4d62-8775-017fc5077417">

<br/>

- VGG16과 Xception, mobile_net 세개의 사전 훈련 모델을 이용하여 유사도를 확인해본 결과

  - vgg16 : jellyfish 98.3259%
  - Xception : mixing_bowl 99.9972%
  - mobilenet : shower_curtain 93.9607%

  <details>
      <summary>VGG16모델 유사도 확인 코드보기</summary>
      
        import numpy as np
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions

        model = VGG16()
        image = load_img('./datasets/jellyfish/train/barrel_jellyfish/01.jpeg', target_size=(224, 224))
        image = img_to_array(image)

        # 불러온 이미지의 차원을 1차수 늘려준다. 4차원으로.
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        target = decode_predictions(prediction)
        print(target)

        # 가장 높은 하나의 답만 출력할 때
        print(target[0][0])
        # 답의 이름과 확률을 출력할 때
        print(target[0][0][1], f'{np.round(target[0][0][2] * 100, 4)}%')

  </details>

<br/>

- 이와같이 vgg16모델이 jellyfish 약 98%로 가장 유사도가 높다고 판단하여 vgg16모델을 사용하여 진행하였습니다.
