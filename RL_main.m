function [states, action, total_reward, Q, time] = RL_main()
    task1 = load('task1.mat');
    reward = reshape(task1.reward, [10, 10, 4]);
    gamma = 0.95;
    Q = zeros(10, 10, 4);
    Q(1, :, 1) = -1;
    Q(10, :, 3) = -1;
    Q(:, 1, 4) = -1;
    Q(:, 10, 2) = -1;
    state_change = [[-1, 0]; [0, 1]; [1, 0]; [0, -1]];
    start_time = datevec(now);
    for trial = 1:3000
%         display(strcat('trial: ', num2str(trial)));
        current_state = [1, 1];
        total_change = 0;
        k = 1;
        while 1
%             display(strcat('trial: ', num2str(trial), ' k: ', num2str(k)));
            alpha = 100 / (100 + k);
            if alpha < 0.005
                break;
            end
            epsilon = alpha;
            current_Q = reshape(Q(current_state(1), current_state(2), :), [1, 4]);
            probability = -1 * ones(4, 1);
            probability(current_Q == -1) = 0;
            if (epsilon >= 1) || (range(current_Q(current_Q ~= -1)) == 0)
                probability(probability==-1) = 1 / sum(current_Q ~= -1);
            else
                [~, I] = max(current_Q);
                probability(I) = 1 - epsilon;
                probability(probability==-1) = epsilon / (sum(current_Q ~= -1) - 1);
            end
            
            current_action = randsample(1:4, 1, true, probability);
            next_reward = reward(current_state(1), current_state(2), current_action);
            next_state = current_state + state_change(current_action, :);
            next_state_Q_max = max(Q(next_state(1), next_state(2), :));
            current_state_Q = Q(current_state(1), current_state(2), current_action);
            delta = alpha * (next_reward + gamma * next_state_Q_max - current_state_Q);
            Q(current_state(1), current_state(2), current_action) = current_state_Q + delta;
            total_change = total_change + abs(delta);
            if next_state == [10, 10]
                break;
            end
            current_state = next_state;
            k = k + 1;
        end
        if total_change < 1e-6
            break;
        end
    end
    end_time = datevec(now);
    time = etime(end_time, start_time);
    hold on;
    for i=0:10
        plot([0, 10], [i, i], 'b-');
        plot([i, i], [0, 10], 'b-');
    end
    action_plot = '^>V<';
    states = zeros(19, 2);
    states(1, :) = [1, 1];
    action = zeros(18, 1);
    total_reward = 0;
    for i = 1:18
        [~, I] = max(Q(states(i, 1), states(i, 2), :));
        action(i) = I;
        text(states(i, 2)-0.6, 10.5-states(i, 1), action_plot(action(i)), 'Color','red', 'FontSize', 15);
        states(i+1, :) = states(i, :) + state_change(action(i), :);
        total_reward = total_reward + reward(states(i, 1), states(i, 2), action(i)) * gamma ^ (i - 1);
    end
    text(5.5, 8.5, strcat('total reward: ', num2str(total_reward)), 'Color', 'magenta', 'FontSize', 15);
    hold off;
    set(gca,'xtick',[])
    set(gca,'ytick',[])
end