%  模型评估
disp('=== 开始模型评估 ===');

% 确保测试集特征和标签已正确定义
if ~exist('XTest', 'var') || ~exist('YTest', 'var')
    error('测试集数据未定义，请先完成特征提取步骤');
end

% 使用模型进行预测
testPred = predict(svmModel, XTest);  % 确保svmModel已定义

% 检查预测结果和标签的一致性
if numel(YTest) ~= numel(testPred)
    error('预测结果与标签数量不匹配');
end

% 计算准确率
accuracy = mean(testPred == YTest);
disp(['测试集准确率: ', num2str(accuracy*100, '%.2f'), '%']);

% 获取所有类别
classNames = categories(YTest);
numClasses = numel(classNames);

% 使用confusionchart
% 创建基础混淆矩阵图
figure;
cm = confusionchart(YTest, testPred, ...
    'Title', '绝缘子憎水性等级分类混淆矩阵', ...
    'FontSize', 12, ...
    'RowSummary', 'off', ...          % 关闭右侧行统计
    'ColumnSummary', 'off');          % 关闭底部列统计

% 设置归一化方式（可选）
cm.Normalization = 'absolute';        % 显示绝对数值（默认）
% cm.Normalization = 'row-normalized'; % 或显示行归一化百分比

% 自定义颜色和字体
cm.DiagonalColor = [0.1 0.5 0.1];    % 设置对角线颜色（深绿）
cm.OffDiagonalColor = [0.8 0.2 0.2]; % 设置错误分类颜色（红）
cm.FontColor = 'k';                   % 字体颜色黑色
cm.GridVisible = 'off';               % 关闭网格线（可选）

% 调整坐标轴标签
xlabel('预测标签');
ylabel('真实标签');
set(gca, 'FontSize', 12);             % 设置坐标轴字体大小
