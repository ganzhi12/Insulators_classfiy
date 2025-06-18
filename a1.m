% 设置数据集路径
trainPath = 'D:\dataset\train';
testPath = 'D:\dataset\test';

% 创建图像数据存储对象
imdsTrain = imageDatastore(trainPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest = imageDatastore(testPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 检查类别数量
categories = categories(imdsTrain.Labels);
numClasses = numel(categories);
disp(['发现 ', num2str(numClasses), ' 个憎水性等级类别']);
