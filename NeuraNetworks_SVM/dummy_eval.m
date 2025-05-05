clc;
clear;
close all;

% Load test.mat
load('test.mat');

% % Get the number of samples of test_data
N = size(test_data, 2);

% Randomly select 600 non-repeated indices
rng(71); 
indices = randperm(N, 600);

% Extract labels and data
eval_data = test_data(:, indices);
eval_label = test_label(indices);

% Save as "eval.mat"
save('eval.mat', 'eval_data', 'eval_label');

fprintf("600 data entries have been successfully extracted from test.mat and saved in eval.mat\n");
