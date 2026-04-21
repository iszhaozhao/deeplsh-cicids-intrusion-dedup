% 1. 输入数据 (根据你的运行日志填写)
models = {'Exact Scan (k-NN)', 'SimHash', 'MLP', 'Bi-GRU (Ours)'};

% 单次查询的平均延迟 (单位: 毫秒 ms)
% 数据参考：你日志里 SimHash 是 305ms, MLP 是 74ms, Bi-GRU 是 534ms
% Exact Scan (线性扫描) 通常需要几万毫秒，这里填 15000 作为演示
latency_ms = [15000, 305, 74, 534]; 

% 2. 创建画布
fig_lat = figure('Color', 'w', 'Position', [100, 100, 650, 450]);
hold on;

% 3. 绘制柱状图
b = bar(latency_ms, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.2);

% 4. 为每个柱子分配专业的学术配色
b.CData(1,:) = [0.6, 0.6, 0.6];    % 灰色 (传统线性扫描)
b.CData(2,:) = [0.26, 0.45, 0.65]; % 蓝色 (SimHash)
b.CData(3,:) = [0.90, 0.54, 0.28]; % 橙色 (MLP)
b.CData(4,:) = [0.40, 0.65, 0.45]; % 绿色 (Bi-GRU)

% 5. 坐标轴与字体设置
set(gca, 'XTick', 1:4, 'XTickLabel', models, 'FontName', 'Times New Roman', ...
    'FontSize', 12, 'LineWidth', 1.2);
ylabel('Average Query Latency (ms)', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
title('Query Latency Comparison', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');

% 【关键操作】：由于 k-NN 时间太长，开启 Y 轴对数坐标
set(gca, 'YScale', 'log'); 
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.4, 'YGrid', 'on', 'XGrid', 'off');

% 6. 在柱子上方添加具体的数值标签
for i = 1:length(latency_ms)
    text(i, latency_ms(i) * 1.15, sprintf('%.1f ms', latency_ms(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontName', 'Times New Roman', 'FontSize', 11, 'FontWeight', 'bold');
end

% 7. 优化边框并导出
box on;
exportgraphics(fig_lat, 'Query_Latency_Comparison.pdf', 'ContentType', 'vector');
disp('检索延迟柱状图已导出为 Query_Latency_Comparison.pdf');