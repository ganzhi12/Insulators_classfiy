% 设置数据集路径
trainPath = 'D:\dataset\train';
testPath = 'D:\dataset\test';

% 创建图像数据存储对象
imdsTrain = imageDatastore(trainPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest = imageDatastore(testPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 定义特征提取参数
colorBins = 32;      % 颜色直方图的bin数量
lbpNeighbors = 8;    % LBP邻域点数
lbpRadius = 2;       % LBP半径
hogCellSize = [32 32]; % HOG单元格大小

% 先提取一个样本确定各特征维度
sampleImg = readimage(imdsTrain, 1);
if size(sampleImg, 3) == 3
    sampleGray = rgb2gray(sampleImg);
else
    sampleGray = sampleImg;
end

% 计算各特征维度
colorFeatDim = colorBins * 3;  % 固定为96维 (32x3)
lbpFeat = extractLBPFeatures(sampleGray, 'NumNeighbors', lbpNeighbors, 'Radius', lbpRadius);
lbpFeatDim = length(lbpFeat);
hogFeat = extractHOGFeatures(sampleGray, 'CellSize', hogCellSize);
hogFeatDim = length(hogFeat);
totalFeatDim = colorFeatDim + lbpFeatDim + hogFeatDim;

% 打印特征维度信息
fprintf('特征维度信息:\n');
fprintf('颜色特征: %d 维\n', colorFeatDim);
fprintf('LBP特征: %d 维\n', lbpFeatDim);
fprintf('HOG特征: %d 维\n', hogFeatDim);
fprintf('总特征维度: %d 维\n', totalFeatDim);

% 初始化训练集特征矩阵
trainFeatures = zeros(numel(imdsTrain.Files), totalFeatDim);
trainLabels = imdsTrain.Labels;

% 提取训练集特征
for i = 1:numel(imdsTrain.Files)
    try
        img = readimage(imdsTrain, i);
        
        % 处理颜色特征
        if size(img, 3) == 3
            % RGB图像
            colorFeat = [];
            for ch = 1:3
                hist = imhist(img(:,:,ch), colorBins);
                colorFeat = [colorFeat; hist];
            end
            grayImg = rgb2gray(img);
        else
            % 灰度图像 - 复制三个通道保持维度一致
            colorFeat = [];
            hist = imhist(img, colorBins);
            colorFeat = repmat(hist, 3, 1);  % 复制为3通道
            grayImg = img;
        end
        
        % 确保颜色特征维度正确
        if length(colorFeat) ~= colorFeatDim
            error('颜色特征维度不正确: 期望 %d, 实际 %d', colorFeatDim, length(colorFeat));
        end
        
        % 提取LBP特征
        lbpFeat = extractLBPFeatures(grayImg, 'NumNeighbors', lbpNeighbors, 'Radius', lbpRadius);
        if length(lbpFeat) ~= lbpFeatDim
            error('LBP特征维度不正确: 期望 %d, 实际 %d', lbpFeatDim, length(lbpFeat));
        end
        
        % 提取HOG特征
        hogFeat = extractHOGFeatures(grayImg, 'CellSize', hogCellSize);
        if length(hogFeat) ~= hogFeatDim
            error('HOG特征维度不正确: 期望 %d, 实际 %d', hogFeatDim, length(hogFeat));
        end
        
        % 合并特征
        trainFeatures(i,:) = [colorFeat', lbpFeat, hogFeat];
        
        if mod(i, 10) == 0
            fprintf('已处理 %d/%d 训练图像\n', i, numel(imdsTrain.Files));
        end
    catch ME
        fprintf('处理训练图像 %d 时出错: %s\n', i, ME.message);
        fprintf('图像路径: %s\n', imdsTrain.Files{i});
        rethrow(ME);
    end
end

% 同样的方法处理测试集
testFeatures = zeros(numel(imdsTest.Files), totalFeatDim);
testLabels = imdsTest.Labels;

for i = 1:numel(imdsTest.Files)
    try
        img = readimage(imdsTest, i);
        
        % 处理颜色特征
        if size(img, 3) == 3
            colorFeat = [];
            for ch = 1:3
                hist = imhist(img(:,:,ch), colorBins);
                colorFeat = [colorFeat; hist];
            end
            grayImg = rgb2gray(img);
        else
            colorFeat = [];
            hist = imhist(img, colorBins);
            colorFeat = repmat(hist, 3, 1);
            grayImg = img;
        end
        
        % 提取LBP特征
        lbpFeat = extractLBPFeatures(grayImg, 'NumNeighbors', lbpNeighbors, 'Radius', lbpRadius);
        
        % 提取HOG特征
        hogFeat = extractHOGFeatures(grayImg, 'CellSize', hogCellSize);
        
        % 合并特征
        testFeatures(i,:) = [colorFeat', lbpFeat, hogFeat];
        
    catch ME
        fprintf('处理测试图像 %d 时出错: %s\n', i, ME.message);
        fprintf('图像路径: %s\n', imdsTest.Files{i});
        rethrow(ME);
    end
end

% 标准化特征
[trainFeatures, mu, sigma] = zscore(trainFeatures);
testFeatures = (testFeatures - mu) ./ sigma;

disp('特征提取完成，维度验证:');
disp(['训练集: ', num2str(size(trainFeatures))]);
disp(['测试集: ', num2str(size(testFeatures))]);
