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
    
    % 创建等级说明映射
    levelDescription = containers.Map({
        'HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6'
    }, {
        '完全憎水，水珠清晰', ...
        '憎水，水珠较清晰', ...
        '弱憎水，水珠开始融合', ...
        '中间状态，部分表面湿润', ...
        '弱亲水，大面积湿润', ...
        '亲水，完全湿润'
    });
    
    % 获取当前预测等级的说明
    if levelDescription.isKey(predLabel)
        description = levelDescription(predLabel);
    else
        description = '未知等级';
    end
    
    % 定义不同等级的颜色
    levelColors = containers.Map({
        'HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6'
    }, {
        [0, 0.8, 0],     % 绿色
        [0.5, 0.8, 0],   % 黄绿色
        [1, 1, 0],       % 黄色
        [1, 0.65, 0],    % 橙色
        [1, 0, 0],       % 红色
        [0.5, 0, 0]      % 深红色
    });
    
    % 获取当前预测等级的颜色
    if levelColors.isKey(predLabel)
        labelColor = levelColors(predLabel);
    else
        labelColor = [0, 0, 0]; % 黑色（未知等级）
    end
    
    % 显示原图和结果图
    figure('Position', [100, 100, 1200, 500]);
    
    % 显示原图
    subplot(1, 2, 1);
    imshow(img);
    title('原始图像');
    axis on;
    
    % 创建结果图像
    resultImg = img;
    
    % 在结果图像上添加预测信息
    [height, width, ~] = size(resultImg);
    textSize = max(12, round(min(height, width) / 40));
    
    % 在图像上绘制标签背景
    textPosition = [10, 10];
    textWidth = max(textSize * length(['预测等级: ' predLabel]) * 0.6, textSize * length(['等级说明: ' description]) * 0.6);
    textHeight = textSize * 4;
    
    % 创建一个半透明矩形作为文本背景
    rectangle('Position', [textPosition(1)-5, textPosition(2)-5, textWidth+10, textHeight+10], ...
              'FaceColor', [1 1 1 0.7], 'EdgeColor', 'none');
    
    % 添加预测等级文本
    text(textPosition(1), textPosition(2) + textSize, ['预测等级: ' predLabel], ...
         'Color', labelColor, 'FontSize', textSize, 'FontWeight', 'bold');
    
    % 添加等级说明文本
    text(textPosition(1), textPosition(2) + textSize * 2.5, ['等级说明: ' description], ...
         'Color', [0 0 0], 'FontSize', textSize-2);
    
    % 显示结果图像
    subplot(1, 2, 2);
    imshow(resultImg);
    title(['预测结果: ' predLabel]);
    axis on;
    
    % 打印预测结果
    fprintf('预测憎水性等级: %s\n', predLabel);
    fprintf('等级说明: %s\n', description);
end