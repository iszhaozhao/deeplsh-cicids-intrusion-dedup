% 1. 准备数据 (请确保替换为你实际提取的 true_sim 和 pred_sim)
% 这里假设你已经从 CSV 中提取了这两个变量，比如：
% data = readtable('pairs_validation_bigru.csv');
% true_sim = data.target_similarity;
% pred_sim = data.token_jaccard; % 注意：如果是真实预测值，请用预测列

% --- 仅为演示，如果你的数据读入正常，请删除下面两行随机生成 ---
true_sim = rand(10000, 1); 
pred_sim = true_sim .* (0.8 + 0.2*rand(10000, 1)); 
% -----------------------------------------------------------

% 2. 创建高清晰度画布
fig1 = figure('Color', 'w', 'Position', [100, 100, 650, 500]);
hold on;

% 3. 绘制散点 (使用深蓝色，极低透明度展现数据密度)
% 参数 8 是点的大小，可以根据数据量微调
scatter(true_sim, pred_sim, 8, [0.1, 0.3, 0.6], 'filled', ...
    'MarkerFaceAlpha', 0.05, 'MarkerEdgeAlpha', 0.05);

% 4. 绘制参考线 (深红色虚线，稍粗)
plot([0, 1], [0, 1], '--', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2);

% 5. 坐标轴与文字装饰 (严格的学术规范：Times New Roman)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.2);
xlabel('Target Similarity (Ground Truth)', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Predicted Collision Probability', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
title('Locality-Sensitive Property: Bi-GRU', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');

% 6. 网格与边框优化
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.4); % 淡化网格线
box on; % 开启全封闭边框
axis square; % 强制为正方形，对角线才准确
axis([0 1 0 1]);

% 7. 添加统计文本 (计算 Kendall tau)
tau = corr(true_sim, pred_sim, 'type', 'Kendall');
text(0.05, 0.92, sprintf('Kendall \\tau = %.4f', tau), ...
    'FontName', 'Times New Roman', 'FontSize', 13, 'Color', [0.8, 0.2, 0.2], 'FontWeight', 'bold');

hold off;

% 8. 一键导出高清矢量图 (可以直接插入 Word 或 LaTeX)
exportgraphics(fig1, 'Figure_4_Scatter.pdf', 'ContentType', 'vector');
disp('散点图已成功导出为 Figure_4_Scatter.pdf');