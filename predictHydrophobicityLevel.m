function predLabel = predictHydrophobicityLevel(imagePath)
%PREDICTHYDROPHOBICITYLEVEL 绝缘子憎水性等级识别
%   输入：
%       imagePath - 绝缘子图片文件路径（字符串）
%   输出：
%       predLabel - 预测的憎水性等级

    % 检查文件是否存在
    if ~exist(imagePath, 'file')
        error('文件不存在: %s', imagePath);
    end
    
    % 加载预训练模型和参数
    try
        modelFile = 'insulator_model.mat';
        load(modelFile, 'svmModel', 'mu', 'sigma', ...
             'colorBins', 'lbpNeighbors', 'lbpRadius', 'hogCellSize');
    catch
        error('无法加载模型文件 %s，请确保已训练模型', modelFile);
    end
    
    % 读取图像
    try
        img = imread(imagePath);
        if isempty(img)
            error('无法读取图像文件');
        end
    catch
        error('不支持的图像格式或文件已损坏');
    end
    
    % 图像预处理
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
        img = repmat(img, [1 1 3]); % 灰度图转为伪RGB
    end
    
    % 特征提取
    colorFeat = [];
    for ch = 1:3
        hist = imhist(img(:,:,ch), colorBins);
        colorFeat = [colorFeat; hist]; %#ok<AGROW>
    end
    
    lbpFeat = extractLBPFeatures(grayImg, ...
                               'NumNeighbors', lbpNeighbors, ...
                               'Radius', lbpRadius);
    
    hogFeat = extractHOGFeatures(grayImg, 'CellSize', hogCellSize);
    
    % 特征合并和标准化
    features = [colorFeat', lbpFeat, hogFeat];
    features = (features - mu) ./ sigma;
    
    % 预测等级
    predLabel = char(predict(svmModel, features));
    
    % 显示结果（仅显示等级）
    fprintf('预测憎水性等级: %s\n', predLabel);
end