% 转换为适合SVM的格式
XTrain = trainFeatures;
YTrain = trainLabels;
XTest = testFeatures;
YTest = testLabels;

% 训练多类SVM分类器（使用一对一策略）
t = templateSVM('KernelFunction', 'rbf', 'Standardize', false, ...
                'KernelScale', 'auto', 'BoxConstraint', 1);

% 使用5折交叉验证评估模型
cvModel = fitcecoc(XTrain, YTrain, 'Learners', t, ...
                  'Coding', 'onevsone', 'KFold', 5);

% 计算交叉验证准确率
cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'ClassifError');
disp(['交叉验证准确率: ', num2str(cvAccuracy*100), '%']);

% 训练最终模型（使用全部训练数据）
svmModel = fitcecoc(XTrain, YTrain, 'Learners', t, ...
                   'Coding', 'onevsone', 'Verbose', 1);

% 保存模型和标准化参数
save('insulator_model.mat', 'svmModel', 'mu', 'sigma', ...
     'colorBins', 'lbpNeighbors', 'lbpRadius', 'hogCellSize');
