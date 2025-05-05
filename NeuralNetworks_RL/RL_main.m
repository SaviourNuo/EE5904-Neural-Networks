clc;
clear;
close all;

output_dir = 'task2_clip';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Parameter Settings
load("qeval.mat");
num_states = 100; % Each cell represents a possible state
num_actions = 4; % The robot has for possible actions: 1=up, 2=right, 3=down, 4=left
max_trials = 3000; % The maximum number of trials in each run set to 3000
max_steps = 1000; % The maximum number of steps per trial in each run set to 1000
gamma = 0.9; % Discount rate set to 0.9
Q_threshold = 0.05; % The minimum tolerance of Q value against the last trial 
param_func = @(k) exp(-0.001*k);
% k: Index of the current trial  param_func: e^(-0.001k) as the exploration/learning rate

%% Q-learning training and evaluation for current epsilon and gamma setting, visualization on optimal policy and path
eps_alpha_k = param_func;
success_count = 0; % Record the number of times the goal state is reached
total_time = 0; % The average program execution time of the “goal-reaching” runs
epsilon_threshold = eps_alpha_k(max_trials); % The minimum epsilon after 3000 trials
best_reward = -Inf;
best_policy_actions = [];
best_Q = [];

for run = 1:10 % Run the program 10 times
    tic; % Start the timer
    Q = zeros(num_states, num_actions); % Initialize the Q-table
    
    for trial = 1:max_trials
        s = 1; % Reset the robot to the start state
        steps = 0; 
        Q_old = Q;
        epsilon = eps_alpha_k(trial); % Exploration rate
        alpha = epsilon; % Learning rate
       
        if epsilon < epsilon_threshold * 1.1 % If epsilon is small enough, jump out of the loop
            break;
        end

        while s ~= 100 && steps <= max_steps % Run when the robot hasn't made it to the goal state or been stuck                  

            % Perform epsilon-greedy strategy
            % In each action selection, there is a probability of exploration of ε and exploitation of 1-ε
            if rand < epsilon
                a = randi(num_actions); % Exploration: Randomly select among 1~4
            else
                [~, a] = max(Q(s, :)); % Exploitation: Choose the action with the largest Q at the currernt state and get the corresponding action a
            end

            s_next = get_next_state(s, a); % Get the next state after taking an action
            r = qevalreward(s, a); % Get the reward of corresponding action at the current state
            Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s_next, :)) - Q(s, a)); % Core strategy of Q-learning, update the Q-table
            s = s_next; % Move the robot to the next state
            steps = steps + 1; % Count the steps

        end

        if max(abs(Q(:) - Q_old(:))) < Q_threshold % If the largest difference between all Q values and the corresponding ones ​​obtained in the last trial is less than 0.05, jump out of the loop       
            break;
        end
    end

    [policy, success, path, total_reward] = evaluate_policy(Q, qevalreward);
    %{
        total_reward: The reward of the strategy evaluated in the current run
        best_reward: The best reward over all runs under the current epsilon and gamma setting
    %}

    if success
        success_count = success_count + 1;
        total_time = total_time + toc;

        if total_reward > best_reward
            best_reward = total_reward;
            best_policy_actions = policy(path(1:end-1));
            best_Q = Q;
            best_policy = policy;
        end
    end
end

if best_reward > -Inf
    drawOptPath(best_reward, best_policy_actions);
    drawOptPol(best_Q, best_reward);
    save('task2_clip/optimal_policy_epsilon_exponential_gamma0.9.mat', 'best_policy');
    qevalstates = path(:);
    save('task2_clip/qevalstates.mat', 'qevalstates');
end

avg_time = total_time / max(success_count, 1); % Record the average execution time of a run of complete training
fprintf('Epsilon = exp(-0.001k), gamma = 0.9: %d successes, avg time = %.5f sec\n', success_count, avg_time);

%% Function to find the next state given current state and action
function s_next = get_next_state(s, a)
    [row, col] = ind2sub([10, 10], s); % ind2sub receives the shape of the grade and the current state, gives the corresponding row and column

    switch a
        case 1, row = max(row - 1, 1);
        case 2, col = min(col + 1, 10);
        case 3, row = min(row + 1, 10);
        case 4, col = max(col - 1, 1);
    end % Use 'min' and 'max' to function as borders

    s_next = sub2ind([10, 10], row, col); % Reverse and get the next state
end

%% Function to get the policy and path based on the given Q-table and reward
function [policy, success, path, total_reward] = evaluate_policy(Q, reward)
%{
    Input:
        Q: The Q-table currently learned
        reward: the immediate reward table given
    
    Output:
        policy: The optimal action strategy extracted from the Q table (100*1)
        success: Whether the goal state is successfully reached
        path: The sequence of the states passed
        total_reward: The total reward obtained by reaching the goal state
%}

    num_states = 100;
    policy = zeros(num_states, 1);

    % For each state, select the action with the largest Q value from Q(s,:) as the optimal strategy in that state
    for s = 1:num_states
        [~, a] = max(Q(s, :));
        policy(s) = a;
    end

    s = 1;
    path = s;
    total_reward = 0;
    success = false;

    for step = 1:100
        a = policy(s); % Choose the action for current state according to the policy
        s_next = get_next_state(s, a); % Move to the next state
        total_reward = total_reward + reward(s, a); % Add up the reward
        path(end + 1) = s_next; % Add the next state into path 

        if s_next == 100
            success = true; % The robot has reached the goal state
            break;
        end

        s = s_next; % Update
    end
end

%% Function to plot the path trajectory 
function drawOptPath(total_reward, path_actions)
    pos = 1; % Reset the robot to the start state
    direction = ['^r'; '>r'; 'vr'; '<r']; % Use red triangles to indicate the next action at the current state

    figure;
    hold on;

    for i = 0:10 % Draw a blue border line of a 10*10 grid
        plot([0, 10], [i, i], 'b');
        plot([i, i], [0, 10], 'b');
    end

    plot(9.5, 9.5, '*r'); % Use a red star to mark the goal state
    axis([0 10 0 10]);
    title(['Optimal path (\epsilon = exp(-0.001k), \gamma = 0.9, reward =', num2str(total_reward), ')']);
    grid off;
    set(gca, 'YDir', 'reverse'); % Let the upper left corner of the coordinates be (1, 1)

    offset = [-1, 10, 1, -10];
    
    for i = 1:length(path_actions)
        [row, col] = ind2sub([10, 10], pos);
        x = col - 0.5;
        y = row - 0.5;
        action = path_actions(i);
        plot(x, y, direction(action, :)); % Draw the direction of the current triangle according to the action
        pos = pos + offset(action); % Get the next state according to the action
    end

    hold off;
    filename = sprintf('task2_clip/Path_epsilon_exponential_gamma0.9.png');
    saveas(gcf, filename);
end

%% Function to plot the optimal policy
function drawOptPol(Q, total_reward)
    [~, opt_act] = max(Q, [], 2); % Look for the action conrresponding to the maximum Q value along each row (each row stores the Q value of 4 actions at the current state)
    direction = ['^r'; '>r'; 'vr'; '<r'];

    figure;
    hold on;

    for i = 0:10
        plot([0, 10], [i, i], 'b');
        plot([i, i], [0, 10], 'b');
    end

    plot(9.5, 9.5, '*r');
    axis([0 10 0 10]);
    title(['Optimal policy (\epsilon = exp(-0.001k), \gamma = 0.9, reward =', num2str(total_reward), ')']);
    grid off;
    set(gca, 'YDir', 'reverse');

    for i = 1:length(opt_act)
        [row, col] = ind2sub([10, 10], i);
        x = col - 0.5;
        y = row - 0.5;
        plot(x, y, direction(opt_act(i), :));
    end

    hold off;
    filename = sprintf('task2_clip/Policy_epsilon_exponential_gamma0.9.png');
    saveas(gcf, filename);
end
