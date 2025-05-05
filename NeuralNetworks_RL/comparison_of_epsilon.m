clc;
clear;
close all;

% The value range of k
k = 1:400;

% Define different exploration rate functions
eps1 = 1 ./ k;
eps2 = 100 ./ (100 + k);
eps3 = (1 + log(k)) ./ k;
eps4 = (1 + 5*log(k)) ./ k;
eps5 = exp(-0.001 .* k);

% Plot and save the curves
figure;
plot(k, eps1, 'b', 'LineWidth', 1.5); hold on;
plot(k, eps2, 'r', 'LineWidth', 1.5);
plot(k, eps3, 'm', 'LineWidth', 1.5);
plot(k, eps4, 'k', 'LineWidth', 1.5);
plot(k, eps5, 'g', 'LineWidth', 1.5);

legend({'1/k', '100/(100 + k)', '(1 + log(k))/k', '(1 + 5*log(k))/k', ...
        'exp(-0.001k)'}, ...
        'Location', 'northeast');

xlabel('Iteration k');
ylabel('Exploration / Learning Rate');
title('Comparison of \epsilon_k Functions');
grid on;

filename = sprintf('Comparison of Epsilon Functions.png');
saveas(gcf, filename);