% 1. 输入数据 (来自你之前生成的 JSON 文件)
TN = 59970;  FP = 30;
FN = 104;    TP = 59896;

% 构造混淆矩阵 [真实为负, 真实为正]
cm = [TN, FP; FN, TP];
labels = {'Non-Duplicate (Negative)', 'Duplicate (Positive)'};

% 2. 创建高清晰度画布
fig3 = figure('Color', 'w', 'Position', [300, 300, 550, 450]);

% 3. 绘制热力图矩阵
imagesc(cm);

% 4. 采用高级学术配色 (类似 Seaborn 的蓝白色调)
cmap = [linspace(1, 0.1, 256)', linspace(1, 0.3, 256)', linspace(1, 0.6, 256)'];
colormap(cmap);
cb = colorbar;
cb.FontName = 'Times New Roman';
cb.FontSize = 11;

% 5. 坐标轴设置
set(gca, 'XTick', 1:2, 'XTickLabel', labels, 'YTick', 1:2, 'YTickLabel', labels, ...
    'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.2);
xlabel('Predicted Class', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('True Class', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
title('Confusion Matrix: Bi-GRU DeepLSH', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');

% 6. 在方格中心填写数值
for i = 1:2
    for j = 1:2
        val = cm(i, j);
        % 如果数值很大，用白色字；数值小，用黑色字
        if val > 30000
            textColor = 'w';
        else
            textColor = 'k';
        end
        % 添加带有千位分隔符的文本
        text(j, i, num2str(val, '%d'), 'HorizontalAlignment', 'center', ...
            'Color', textColor, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
    end
end

% 7. 优化边框
axis square;
set(gca, 'XAxisLocation', 'bottom');

% 8. 导出高清图
exportgraphics(fig3, 'Confusion_Matrix.pdf', 'ContentType', 'vector');
disp('混淆矩阵已成功导出为 Confusion_Matrix.pdf');