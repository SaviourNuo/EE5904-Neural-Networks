function [g, model, K_train] = svm_discriminant(data_train, data_test, label, C, p, kernel_type, model_in)

%{
    Input:
        data_train: Training data matrix (d × N)
        data_test: Test data matrix (d × M)
        label: Label vector (N × 1) for training data, with values in {-1, 1}
        C: Regularization parameter (soft-margin penalty term); use large value for hard-margin
        p: Degree of the polynomial kernel
        kernel_type: Type of kernel used in SVM
        model_in: Optional. If provided, prediction will be made using this pretrained model (skip training phase)
    Output:
        g:  Discriminant function values (1 × M)
        model: Struct containing SVM model parameters
        K_train: Kernel (Gram) matrix used during training, useful for checking kernel admissibility
%} 

    % === Define different SVM with possible kernels ===
    % If model_in is not passed in, go through the training process
    if nargin < 7 || isempty(model_in) % Check the number of arguments and if the model is passed in
        N = size(data_train, 2); % Get the number of samples of train_data
        f = -ones(N, 1); % Predefine for later use of the linear term "-sum(ai)" (function to minimize "0.5a' * H * a -sum(a_i)")
        A = []; b = []; % No inequality constraints for SVM (meaning no Ax <= b constraint)
        Aeq = label'; % Equality constraint Aeq * a = beq
        beq = 0; % sum(a_i * y_i) = 0 ensures "the force of the two types of the samlpes on the hyperplane is the same"
        lb = zeros(N, 1); % Lower bound: a_i >= 0（Lagrange multiplier）
        ub = C * ones(N, 1); % Upper bound
        x0 = []; % No initial point given
        
        switch kernel_type
            case 'linear'
                K_train = data_train' * data_train; % Define an N*N linear kenel, Gram matrix K(x_i, x_j) = x_i' * x_j
            case 'polynomial'
                K_train = (data_train' * data_train + 1).^p; % K(x_i, x_j) = (x_i' * x_j + 1)^p
            otherwise
                error("Unknown kernel type.");
        end

        H = label * label' .* K_train; % Define an N*N symmetric matrix H_ij = y_i * y_j' .* K(x_i, x_j) for the dual form of SVM
        H = (H + H') / 2; % To ensure symmetry and positive definiteness (to avoid numerical errors), force H to be symmetric

        options = optimoptions('quadprog', 'Display', 'Iter', 'MaxIterations', 1000); 
        [alpha, ~, exitflag] = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options); % Use quadprog and display iteration information

        %{

            Input:
                quadprog: Function to solve quadratic programming
                H: Matrix of Quadratic term
                f: Vector of linear term 
                A,b: Inequality constraint
                Aeq,beq: Equality constraint
                lb,ub: Lower/Upper bound

            Output:
                alpha: Lagrange multipliers of all samples (optimal solution)
                % fval: Minimum value of the funciton (0.5a' * H * a -sum(a_i))
                exitflag: Successful or not (1 for success, 0 or negative number for unsuccess)
        %}

        % === Find the support vectors ===
        threshold = 1e-6;
        sv_idx = find(alpha > threshold); % Locate support vectors (those who actually function for determining the hyperface)
        sv_alpha = alpha(sv_idx); % Get SV
        sv_data = data_train(:, sv_idx); % Get the features of the selected SV
        sv_label = label(sv_idx); % Get the labels of SV

        switch kernel_type
            case 'linear'
                K_sv = data_train' * sv_data; % N x M
            case 'polynomial'
                K_sv = (data_train' * sv_data + 1).^p; % N x M
        end

        g_sv = (alpha .* label)' * K_sv; % 1 x M
        b = mean(sv_label' - g_sv); % Average over support vectors

        % === Pack the model ===
        model = struct('sv_alpha', sv_alpha, ...
                       'sv_label', sv_label, ...
                       'sv_data', sv_data, ...
                       'b', b, ...
                       'p', p, ...
                       'kernel_type', kernel_type, ...
                       'exitflag', exitflag);
    else
        model = model_in;
        K_train = []; % Not used during test
    end

    p = model.p;
    kernel_type = model.kernel_type;
    sv_alpha = model.sv_alpha;
    sv_label = model.sv_label;
    sv_data  = model.sv_data;
    b        = model.b;

    switch kernel_type
        case 'linear'
            K = sv_data' * data_test;
        case 'polynomial'
            K = (sv_data' * data_test + 1).^p;
    end

    g = (sv_alpha .* sv_label)' * K + b; % Calculate the discriminant funcion 1 x N
end
