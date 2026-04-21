% 1. 读取数据
opts = detectImportOptions('pairs_validation_bigru.csv');
data = readtable('pairs_validation_bigru.csv', opts);

% 2. 映射正确的变量名
% 根据你的输出，变量名是 target_similarity
true_sim = data.target_similarity; 

% 注意：如果 CSV 里没有 pred_sim，通常是因为模型预测值在另一个列
% 如果你的 CSV 只有你列出的那 10 列，可能需要重新确认 pred_sim 的来源
% 假设 pred_sim 存在（有时在诊断文件的后续列中），如果不存在请告诉我
try
    pred_sim = data.pred_sim; 
catch
    % 如果没有 pred_sim，我们暂时用 token_jaccard 作为对比演示
    % 在论文中，Jaccard 常被用作基准相似度
    pred_sim = data.token_jaccard; 
    warning('未找到 pred_sim，当前绘图使用 token_jaccard 代替展示样式');
end

% 3. 绘图：复刻 Figure 4 风格
figure('Color', 'w', 'Name', 'DeepLSH Property Verification');
hold on;

% 绘制 50 万条数据的散点 (使用 0.05 的极低透明度防止堆叠)
scatter(true_sim, pred_sim, 3, [0 0.447 0.741], 'filled', ...
        'MarkerFaceAlpha', 0.05, 'MarkerEdgeAlpha', 0.05);

% 绘制 y = x 理想参考线
plot([0 1], [0 1], 'r--', 'LineWidth', 2);

% 图表装饰
xlabel('Target Similarity (Ground Truth)', 'FontSize', 12);
ylabel('Collision Probability / Predicted Similarity', 'FontSize', 12);
title('Locality-Sensitive Property: Bi-GRU', 'FontSize', 14);

grid on;
axis([0 1 0 1]);
set(gca, 'Box', 'on', 'TickDir', 'out', 'LineWidth', 1);

% 4. 计算相关性系数 (对应论文 RQ1)
[tau, pval] = corr(true_sim, pred_sim, 'type', 'Kendall');
text(0.05, 0.9, sprintf('Kendall tau = %.4f', tau), 'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');

hold off;
