# 🚩 Deep learning pretrained model project

## 나비와 나방 이미지 분류

> https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species

### 목차

### 1. 데이터 확인 및 전처리

- 데이터는 총 100종의 나비와 나방이 있습니다.

- 유사도가 높은 사전훈련모델을 찾아서 하려 하였으나 나비와 나방에 관련된 사전 훈련모델이 없다고 판단하여 나비와 나방의 이미지가 복잡하다 판단하여 Xception모델으로 진행하였습니다.

- 데이터의 갯수가 약 14000개정도라서 초기 배치사이즈를 64로 설정 후 진행하였습니다.
  <details>
      <summary>데이터 불러오기 코드보기</summary>
      
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        IMAGE_SIZE = 112
        BATCH_SIZE = 64

        train_dir = './datasets/butterfly_moth/train'
        validation_dir = './datasets/butterfly_moth/valid'
        test_dir = './datasets/butterfly_moth/test'

        train_data_generator = ImageDataGenerator(rescale=1./255)
        validation_data_generator = ImageDataGenerator(rescale=1./255)
        test_data_generator = ImageDataGenerator(rescale=1./255)

        train_generator = train_data_generator.flow_from_directory(
            train_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        validation_generator = validation_data_generator.flow_from_directory(
            validation_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_generator = test_data_generator.flow_from_directory(
            test_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

  </details>

<br/>

- 이미지는 아래와같이 나비와 나방이 화려하고 독특한 패턴이 많아서 복잡하다고 판단하였습니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/9fb72efd-f605-4d7d-a624-8e70574aa57d">

- 매직메소드 재정의 후 데이터셋을 반환하는 클래스를 만들어 사용하였습니다.

  <details>
    <summary>클래스 코드 보기</summary>

        import numpy as np
        from tensorflow.keras.utils import Sequence
        from sklearn.utils import shuffle
        import cv2

        IMAGE_SIZE = 112
        BATCH_SIZE = 32

        class Dataset(Sequence):
            def __init__(self, file_paths, targets, batch_size=BATCH_SIZE, aug=None, preprocess=None, shuffle=False):
                self.file_paths = file_paths
                self.targets = targets
                self.batch_size = batch_size
                self.aug = aug
                self.preprocess = preprocess
                self.shuffle = shuffle

        if self.shuffle:
            # 에포크 종료 시, 객체 생성 및 데이터 섞기
            self.on_epoch_end()

        # __len__()는 전체 데이터 건수에서 batch_size 단위로 나눈 데이터 수
        # 예를 들어, 1000개의 데이터를 30 batch_size로 설정하면, 1 batch당 33.33..개이다.
        # 이 때, 소수점은 무조건 올려서 33 + 1 = 34개로 설정한다.
        def __len__(self):
            return int(np.ceil(len(self.targets) / self.batch_size))

        # batch_size 단위로 이미지 배열과 타켓 데이터들을 가져온 뒤 변환한 값을 리턴한다.
        def __getitem__(self, index):
            file_paths_batch = self.file_paths[index * self.batch_size: (index + 1) * self.batch_size]
            targets_batch = self.targets[index * self.batch_size: (index + 1) * self.batch_size]

            results_batch = np.zeros((file_paths_batch.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))

            for i in range(file_paths_batch.shape[0]):
                image = cv2.cvtColor(cv2.imread(file_paths_batch[i]), cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

                if self.aug is not None:
                    image = self.aug(image=image)['image']

                if self.preprocess is not None:
                    image = self.preprocess(image)

                results_batch[i] = image

            return results_batch, targets_batch

        def on_epoch_end(self):
            if self.shuffle:
                self.file_paths, self.targets = shuffle(self.file_paths, self.targets)

  </details>

</br>

- 위에서 선언한 클래스를 사용하여 train, test, validation 데이터세트를 만들어주었습니다.

  <details>
    <summary>데이터셋 생성 코드 보기</summary>

        train_dataset = Dataset(train_file_paths,
                          train_targets,
                          batch_size=BATCH_SIZE,
                          preprocess=xception_preprocess_input,
                          shuffle=True)

        validation_dataset = Dataset(validation_file_paths,
                                validation_targets,
                                batch_size=BATCH_SIZE,
                                preprocess=xception_preprocess_input)

        test_dataset = Dataset(test_file_paths,
                                test_targets,
                                batch_size=BATCH_SIZE,
                                preprocess=xception_preprocess_input)

  </details>

</br>

### 2. 모델 생성 및 모델 학습

- 사전훈련모델을 불러와 분류기만 직접 작성하여 사용하는 함수를 통해 모델을 만들어주었습니다.

  <details>
    <summary>데이터셋 생성 코드 보기</summary>

        def create_model(model_name='vgg16', verbose=False):
          input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
          if model_name == 'vgg16':
              model = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
          elif model_name == 'resnet50': # ResNet50, 74.9% ; ResNet50V2, 76.0%
              model = ResNet50V2(input_tensor=input_tensor, include_top=False, weights='imagenet')
          elif model_name == 'xception': # Inception을 기초로 한 모델
              model = Xception(input_tensor=input_tensor, include_top=False, weights='imagenet')
          elif model_name == 'mobilenet':
              model = MobileNetV2(input_tensor=input_tensor, include_top=False, weights='imagenet')

          x = model.output

          # 분류기
          x = GlobalAveragePooling2D()(x)
          if model_name != 'vgg16':
              x = Dropout(rate=0.5)(x)
          x = Dense(50, activation='relu')(x)
          if model_name != 'vgg16':
              x = Dropout(rate=0.5)(x)
          output = Dense(100, activation='softmax', name='output')(x)

          model = Model(inputs=input_tensor, outputs=output)

          if verbose:
              model.summary()

          return model

        model = create_model(model_name='xception', verbose=True)
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['acc'])

  </details>

</br>

- 해당 모델로 학습 진행 결과

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/e12aa6e6-4073-435c-96b5-ba93a4c53a13">

  - 굉장히 양호한 결과가 나타나서 더이상 건들 필요 없다고 판단하였고, 한 에포크당 11분에서 12분씩 걸리기때문에 시간 단축을 위해 미세조정을 진행하였습니다.

### 3. 미세 조정

- 해당 모델과 데이터는 유사도가 낮다고 판단되지만 시간 단축만 해본다는 생각으로 진행하였습니다.

- 유사도가 낮기때문에 최상위 층만 조금 freeze하였습니다.

  <details>
    <summary>미세조정 코드 보기</summary>

        def fine_tune(datas, model_name, preprocess):
        FIRST_EPOCHS = 10
        SECOND_EPOCHS = 10

        train_file_paths, train_targets, \
        validation_file_paths, validation_targets, \
        test_file_paths, test_targets = datas

        train_dataset = Dataset(train_file_paths,
                            train_targets,
                            batch_size=BATCH_SIZE,
                            preprocess=preprocess,
                            shuffle=True)

        validation_dataset = Dataset(validation_file_paths,
                                validation_targets,
                                batch_size=BATCH_SIZE,
                                preprocess=preprocess)

        model = create_model(model_name=model_name, verbose=True)
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['acc'])

        # feature extractor layer들을 전부 freeze
        for layer in model.layers[:-5]:
            layer.trainable = False

        model.fit(train_dataset,
                  batch_size=BATCH_SIZE,
                  epochs=FIRST_EPOCHS,
                  validation_data=validation_dataset)

        # 배치 정규화만 freeze 진행
        for layer in model.layers:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        # 부분 freeze 진행
        for layer in model.layers[:14]:
            layer.trainable = False

        model.compile(optimizer=Adam(0.00001), loss=CategoricalCrossentropy(), metrics=['acc'])
        history = model.fit(train_dataset,
                  batch_size=BATCH_SIZE,
                  epochs=SECOND_EPOCHS,
                  validation_data=validation_dataset)

        return model, history

        model, history = fine_tune((train_file_paths, train_targets,
           validation_file_paths, validation_targets,
           test_file_paths, test_targets),
          'xception',
          xception_preprocess_input)

  </details>

</br>

- 모델 훈련 결과는 아래와 같습니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/7c5eab7c-f0eb-49ec-ad98-ee5f99717110">

- 확실히 기존 Xception 모델을 사용하였을때보다 소요시간이 많이 줄어든 모습을 보입니다.
- 정확도가 0.6정도로 많이 낮아졌습니다.
- 하지만 검증데이터와 테스트 데이터의 정확도 또한 0.8~0.9 정도로 조금 내려간 모습을 보입니다.

### 4. 모델 평가

- 검증과 테스트데이터의 정확도가 높아서 성능을 평가해보기 위해 직접 이미지를 넣어서 예측해본 결과

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/12155918-abb4-488a-b0a0-ff685e92dc5c">

  - 스무개의 이미지 중 두개를 제외하고 모두 맞춘 모습을 보이며 좋은 성능을 보였습니다.

### 5. 느낀점

- 사전 훈련모델의 성능이 아주 뛰어나다는것을 확실하게 느꼈으며, 나비와 나방에 유사도가 없는 Xception 모델을 사용하였음에도 아주 높은 성능을 보였습니다.