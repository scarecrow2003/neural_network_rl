function [states, action, total_reward, Q] = q_learning(gamma, alpha_option)
    task1 = load('task1.mat');
    reward = reshape(task1.reward, [10, 10, 4]);
    Q = zeros(10, 10, 4);
    Q(1, :, 1) = -1;
    Q(10, :, 3) = -1;
    Q(:, 1, 4) = -1;
    Q(:, 10, 2) = -1;
    for trial = 1:3000
        display(strcat('trial: ', num2str(trial)));
        switch alpha_option
            case 1
                max_k = 210; % alpha = 1 / k < 0.005
            case 2
                max_k = 20000; % alpha = 100 / (100 + k) < 0.005
            case 3
                max_k = 780; % alpha = (1 + log(k)) / k < 0.005
            case 4
                max_k = 3800; % alpha = (1 + 5log(k)) / k < 0.005
        end
        current_state = [1, 1];
        total_change = 0;
        for k = 1:max_k
            display(strcat('trial: ', num2str(trial), ' k: ', num2str(k)));
            switch alpha_option
                case 1
                    alpha = 1 / k;
                case 2
                    alpha = 100 / (100 + k);
                case 3
                    alpha = (1 + log(k)) / k;
                case 4
                    alpha = (1 + 5 * log(k)) / k;
            end
            
            current_Q = reshape(Q(current_state(1), current_state(2), :), [1, 4]);
            [~, I] = max(current_Q);
            probability = -1 * ones(4, 1);
            probability(current_Q == -1) = 0;
            probability(I) = 1 - alpha;
            probability(probability==-1) = alpha / (sum(current_Q ~= -1) - 1);
            
            current_action = randsample(1:4, 1, true, probability);
            next_reward = reward(current_state(1), current_state(2), current_action);
            state_change = zeros(1, 2);
            switch current_action
                case 1
                    state_change = [-1, 0];
                case 2
                    state_change = [0, 1];
                case 3
                    state_change = [1, 0];
                case 4
                    state_change = [0, -1];
            end
            next_state = current_state + state_change;
            if next_state == [10, 10]
                break;
            end
            next_state_Q_max = max(Q(next_state(1), next_state(2), :));
            current_state_Q = Q(current_state(1), current_state(2), current_action);
            delta = alpha * (next_reward + gamma * next_state_Q_max - current_state_Q);
            Q(current_state(1), current_state(2), current_action) = current_state_Q + delta;
            total_change = total_change + abs(delta);
            current_state = next_state;
        end
        if total_change < 1e-6
            break;
        end
    end
    states = zeros(19, 2);
    states(1, :) = [1, 1];
    action = zeros(18, 1);
    total_reward = 0;
    for i = 1:18
        [~, I] = max(Q(states(i, 1), states(i, 2), :));
        action(i) = I;
        state_change = zeros(1, 2);
        switch action(i)
            case 1
                state_change = [-1, 0];
            case 2
                state_change = [0, 1];
            case 3
                state_change = [1, 0];
            case 4
                state_change = [0, -1];
        end
        states(i+1, :) = states(i, :) + state_change;
        total_reward = total_reward + reward(states(i, 1), states(i, 2), action(i));
    end
end