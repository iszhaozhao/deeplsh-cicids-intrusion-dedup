% 1. 输入数据 (根据你实际的评估结果替换)
% 顺序：[SimHash, MLP, Bi-GRU]
models = {'SimHash', 'MLP', 'Bi-GRU (Ours)'};
precision = [0.8421, 0.9632, 0.9995]; % 填入你的真实 Precision
recall    = [0.8015, 0.9514, 0.9982]; % 填入你的真实 Recall
f1_score  = [0.8213, 0.9572, 0.9988]; % 填入你的真实 F1

% 组合成一个矩阵 (3个模型 x 3个指标)
data_matrix = [precision; recall; f1_score]'; 

% 2. 创建画布
fig2 = figure('Color', 'w', 'Position', [200, 200, 700, 450]);

% 3. 绘制分组柱状图
b = bar(data_matrix, 'grouped', 'EdgeColor', 'k', 'LineWidth', 1);

% 4. 设置高级学术配色 (莫兰迪色系：蓝、橙、绿)
b(1).FaceColor = [0.26, 0.45, 0.65]; % 沉稳蓝 (Precision)
b(2).FaceColor = [0.90, 0.54, 0.28]; % 柔和橙 (Recall)
b(3).FaceColor = [0.40, 0.65, 0.45]; % 护眼绿 (F1-score)

% 5. 坐标轴与字体设置
set(gca, 'XTickLabel', models, 'FontName', 'Times New Roman', 'FontSize', 13, 'LineWidth', 1.2);
ylabel('Score (0 - 1.0)', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
title('Performance Comparison of Hash Learning Models', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');

% Y轴范围：由于大家分数都挺高，可以把起点设在 0.7 放大差异
ylim([0.7, 1.05]); 
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3, 'YGrid', 'on', 'XGrid', 'off');

% 6. 添加图例
legend({'Precision', 'Recall', 'F1-Score'}, 'Location', 'northwest', 'FontName', 'Times New Roman', 'FontSize', 12);

% 7. 在柱子顶部添加数值标签 (最显专业的一步)
for i = 1:size(data_matrix, 2)
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    labels = string(round(b(i).YData, 4)); % 保留4位小数
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontName', 'Times New Roman', 'FontSize', 10);
end

% 8. 导出高清图
exportgraphics(fig2, 'Performance_BarChart.pdf', 'ContentType', 'vector');
disp('柱状图已成功导出为 Performance_BarChart.pdf');