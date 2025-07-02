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
    
    % 获取图像文件的路径信息
    [folder, ~, ~] = fileparts(imagePath);
    % 提取文件夹名称
    [~, folderName] = fileparts(folder);
    
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
    
    % 创建单个图形窗口
    figure('Position', [100, 100, 1200, 500]);
    
    % 左侧子图：只显示原始图像，不显示任何文字标注
    subplot(1, 2, 1);
    imshow(img);
    title(['原始图像 - 文件夹: ', folderName]);
    axis on;
    
    % 右侧子图：只显示预测的等级文本
    subplot(1, 2, 2);
    cla; % 清除当前轴
    axis off; % 隐藏坐标轴
    
    % 确定文本大小和位置
    textSize = 24; % 增大文本大小以便清晰显示
    textPosition = [0.5, 0.6]; % 文本位置（居中）
    
    % 添加预测等级文本（使用大字体）
    text(textPosition(1), textPosition(2), ['预测等级: ', predLabel], ...
         'Color', labelColor, 'FontSize', textSize+6, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center');
    
    % 添加等级说明文本
    text(textPosition(1), textPosition(2) - 0.15, ['等级说明: ', description], ...
         'Color', [0 0 0], 'FontSize', textSize-2, ...
         'HorizontalAlignment', 'center');
    
    % 添加预测可信度或其他相关信息（示例）
    confidence = 95; % 示例可信度值
    text(textPosition(1), textPosition(2) - 0.3, ['可信度: ', num2str(confidence), '%'], ...
         'Color', [0.5 0.5 0.5], 'FontSize', textSize-6, ...
         'HorizontalAlignment', 'center');
    
    title('预测结果');
    
    % 打印预测结果
    fprintf('预测憎水性等级: %s\n', predLabel);
    fprintf('等级说明: %s\n', description);
end