% 1. 使用实际路径读取
filePath = '/Users/zhao/domo_codex/deep-locality-sensitive-hashing-main/artifacts/cicids/results/full_500k/pairs_validation_mlp.csv';
data = readtable(filePath);

% 2. 修正列名映射
% 真实相似度：CSV里叫 target_similarity
true_sim = data.target_similarity; 

% 预测相似度：如果 pred_sim 还是找不到，先用 token_jaccard 检查绘图逻辑
try
    pred_sim = data.pred_sim; 
catch
    warning('依然没找到 pred_sim，暂时使用 token_jaccard 进行演示');
    pred_sim = data.token_jaccard; 
end

% 3. 创建高清晰度画布
fig1 = figure('Color', 'w', 'Position', [100, 100, 650, 500]);
hold on;

% 4. 绘制散点 (深蓝色，极低透明度展现数据密度)
scatter(true_sim, pred_sim, 8, [0.1, 0.3, 0.6], 'filled', ...
    'MarkerFaceAlpha', 0.05, 'MarkerEdgeAlpha', 0.05);

% 5. 绘制理想的 y = x 对角参考线
plot([0, 1], [0, 1], '--', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2);

% 6. 坐标轴与文字装饰
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.2);
xlabel('Target Similarity (Ground Truth)', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Predicted Collision Probability', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
title('Locality-Sensitive Property: Bi-GRU', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');

% 7. 网格与边框优化
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.4); 
box on; 
axis square; 
axis([0 1 0 1]);

% 8. 计算并添加 Kendall tau 相关系数
tau = corr(true_sim, pred_sim, 'type', 'Kendall');
text(0.05, 0.92, sprintf('Kendall \\tau = %.4f', tau), ...
    'FontName', 'Times New Roman', 'FontSize', 13, 'Color', [0.8, 0.2, 0.2], 'FontWeight', 'bold');

hold off;

% 9. 一键导出高清矢量图 (文件会保存在 MATLAB 当前目录下)
exportgraphics(fig1, 'Figure_4_Scatter_BiGRU_Final.pdf', 'ContentType', 'vector');
disp('【成功】散点图已导出为 Figure_4_Scatter_BiGRU_Final.pdf');