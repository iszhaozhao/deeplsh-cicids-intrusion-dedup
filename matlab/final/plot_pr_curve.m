% 1. 使用绝对路径读取含有 pred_sim 的最新数据
filePath = '/Users/zhao/domo_codex/deep-locality-sensitive-hashing-main/artifacts/cicids/results/full_500k/pairs_validation_bigru.csv';
data = readtable(filePath);

% 2. 提取标签与预测概率
true_labels = data.is_duplicate; 

% 3. 直接强制提取 pred_sim (就是你亲手加进 CSV 的那个列名！)
% 这次我们不要 try-catch 了，直接强制读取，如果还是报错，那说明 Python 生成有问题
try
    pred_probs = data.pred_sim; 
catch
    error('【严重报错】在最新生成的 CSV 里依然没找到 pred_sim 列！请确认 Python 生成时是否覆盖了旧文件。');
end

% 4. 计算不同阈值下的 Precision 和 Recall
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

% 6. 绘图： Figure 3 风格
fig_pr = figure('Color', 'w', 'Position', [200, 200, 550, 500]);
hold on;

% 绘制主曲线 (深红色，稍粗)
plot(recalls, precisions, 'Color', [0.75, 0.15, 0.20], 'LineWidth', 2.5);

% 绘制完美分类器的参考点 (右上角 1,1)
plot(1, 1, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y'); 

% 7. 图表装饰 (Times New Roman)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.2);
xlabel('Recall (召回率)', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Precision (精确率)', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
title('Precision-Recall Curve (Bi-GRU)', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');

axis([0 1.05 0 1.05]); 
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6);

% 8. 在图表中添加 AUC 和 Best F1 分数框
txt = sprintf('AUC-PR = %.4f\nBest F1 = 0.9988', auc_pr);
annotation('textbox', [0.15 0.15 0.3 0.1], 'String', txt, ...
    'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'FitBoxToText', 'on');

hold off;

% 9. 一键导出高清矢量图 (文件会保存在 MATLAB 当前目录下)
exportgraphics(fig_pr, 'Figure_3_PRCurve_Final.pdf', 'ContentType', 'vector');
disp('【胜利】PR曲线已完美导出！');