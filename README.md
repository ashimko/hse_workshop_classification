# Метрики

Ключевой метрикой для оценки качества модели была выбрана метрика ROC AUC, поскольку именно площадь под кривой и то, насколько больше (меньше) значение этой метрики чем 0.5 - показывают насколько качественно была обучена модель и какова разница между предсказанным и истинным значением.

Однако также было интересно взглянуть и на другие метрики для разных моделей: для лучшей, найденной с помощью hyperopt, и для catboost. Во второй моделе такие низкие оценки, поскольку модель пыталась предсказывать все классы, не игнорирую малочисленные, и из-за дисбаланса данных - получалось у нее это достаточно плохо.

| Model         | accuracy | precision | f1     | recall | ROC AUC                                     |
| ------------- | -------- | --------- | ------ | ------ | ------------------------------------------- |
| Best hyperopt | 0.531    | 0.6842    | 0.5693 | 0.4875 | 0.82745, 0.51902, 0.60317, 0.57442, 0.75862 |
| CatBoost      | 0.5      | 0.6323    | 0.5810 | 0.5375 | 0.83573, 0.66033, 0.63591, 0.55814, 0.72414 |
