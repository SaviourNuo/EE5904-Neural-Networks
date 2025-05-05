load('train.mat');
load('test.mat');
load('eval.mat')

%% === Standardization on training, test and evaluation set===
mu = mean(train_data, 2);
sigma = std(train_data, 0, 2);
sigma(sigma == 0) = 1;

x_train = (train_data - mu) ./ sigma;
x_test  = (test_data  - mu) ./ sigma;
x_eval = (eval_data - mu) ./ sigma;
N_train = size(x_train, 2);
N_test  = size(x_test, 2);
N_eval = size(x_eval, 2);

%% === Hyperparameters ===
sigma_rbf = 10;
gamma = 0.5 / sigma_rbf^2;
C = 1000;

%% === Construct Gram Matrix ===
D2 = pdist2(x_train', x_train').^2;
K = exp(-gamma * D2);
K = (K + K') / 2;

%% === Define the SVM with RBF kernel ===
H = (train_label * train_label') .* K;
H = (H + H') / 2;
f = -ones(N_train, 1);
A = []; b = [];
Aeq = train_label'; beq = 0;
lb = zeros(N_train, 1);
ub = ones(N_train, 1) * C;
x0 = [];

options = optimoptions('quadprog', 'Display', 'Iter', 'MaxIterations', 1000);
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options);

%% === Compute bias from margin SVs ===
threshold = 1e-4;
sv_idx = find(alpha > threshold & alpha < C - threshold);
alpha_sv = alpha(sv_idx);
y_sv = train_label(sv_idx);
x_sv = x_train(:, sv_idx);

bias_all = zeros(length(sv_idx), 1);
for i = 1:length(sv_idx)
    xi = x_sv(:, i);
    k_val = exp(-gamma * sum((x_train - xi).^2, 1));  % 1 x N_train
    gx = alpha' .* train_label' .* k_val;
    bias_all(i) = y_sv(i) - sum(gx);
end
bias = mean(bias_all);

%% === Predict function ===
function pred = rbf_predict(X, x_train, alpha, train_label, gamma, bias)
    D2 = pdist2(x_train', X').^2;
    K = exp(-gamma * D2);  % N_train x N_test
    g = (alpha .* train_label)' * K + bias;
    pred = sign(g)';
end

%% === Predict & Evaluate ===
y_train_pred = rbf_predict(x_train, x_train, alpha, train_label, gamma, bias);
train_acc = mean(y_train_pred == train_label) * 100;

y_test_pred = rbf_predict(x_test, x_train, alpha, train_label, gamma, bias);
test_acc = mean(y_test_pred == test_label) * 100;

eval_predicted = rbf_predict(x_eval, x_train, alpha, train_label, gamma, bias);
eval_acc = mean(eval_predicted == eval_label) * 100;

fprintf("\n[Evaluation Results]\n");
fprintf("Train       Accuracy: %.2f%%\n", train_acc);
fprintf("Test        Accuracy: %.2f%%\n", test_acc);
fprintf("Evaluation  Accuracy: %.2f%%\n", eval_acc);

%% === Save Output for Submission ===
save('eval_predicted.mat', 'eval_predicted');
