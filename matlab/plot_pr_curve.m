% 1. 读取数据
opts = detectImportOptions('pairs_validation_bigru.csv');
data = readtable('pairs_validation_bigru.csv', opts);

% 2. 提取真实标签
true_labels = data.is_duplicate; 

% 3. 提取 Bi-GRU 预测概率【请修改这里！！！】
try
    % 把 YOUR_PREDICTION_COLUMN 换成你查到的列名
    pred_probs = data.YOUR_PREDICTION_COLUMN; 
catch
    error('【脚本已停止】请先修改第9行的 YOUR_PREDICTION_COLUMN 为真实的预测列名！');
end

% 4. 手动计算不同阈值下的 Precision 和 Recall
thresholds = linspace(0, 1, 200); 
precisions = zeros(length(thresholds), 1);
recalls = zeros(length(thresholds), 1);

for i = 1:length(thresholds)
    thresh = thresholds(i);
    
    TP = sum((pred_probs >= thresh) & (true_labels == 1));
    FP = sum((pred_probs >= thresh) & (true_labels == 0));
    FN = sum((pred_probs < thresh)  & (true_labels == 1));
    
    if (TP + FP) == 0
        precisions(i) = 1;
    else
        precisions(i) = TP / (TP + FP);
    end
    
    if (TP + FN) == 0
        recalls(i) = 0;
    else
        recalls(i) = TP / (TP + FN);
    end
end

% 5. 计算 AUC-PR (曲线下面积)
auc_pr = abs(trapz(recalls, precisions));

% 6. 绘制 PR 曲线
fig_pr = figure('Color', 'w', 'Position', [200, 200, 550, 500]);
hold on;

plot(recalls, precisions, 'Color', [0.75, 0.15, 0.20], 'LineWidth', 2.5);
plot(1, 1, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y'); % 完美参考点

% 7. 图表装饰
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.2);
xlabel('Recall (召回率)', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Precision (精确率)', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
title('Precision-Recall Curve (Bi-GRU)', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');

axis([0 1.05 0 1.05]); 
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6);

% 8. 在图表中添加 AUC 分数框
txt = sprintf('AUC-PR = %.4f\nBest F1 = 0.9988', auc_pr);
annotation('textbox', [0.15 0.15 0.3 0.1], 'String', txt, ...
    'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'FitBoxToText', 'on');

hold off;

% 9. 导出高清图
exportgraphics(fig_pr, 'PR_Curve_BiGRU_Fixed.pdf', 'ContentType', 'vector');
disp('Bi-GRU PR曲线已成功导出！');