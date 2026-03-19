%% Auditory Cortex Classification Script (Stim vs Control - Left and Right Cortex Separately)

%% Parameters
Fs = 625;
prest = 500;
post = 1000;
target_length = 2048;
voices_per_octave = 40;
freq_limits = [3 40];
num_top_features = 100;
explained_variance_threshold = 98;

%% Load data
load stim_times
load subj3_left subj3_right subj3_control_left subj3_control_right

%% Segment Trials
[ltrials, t] = signal_to_trials(subj3_left, stim_times, prest, post);
[rtrials, ~] = signal_to_trials(subj3_right, stim_times, prest, post);
[cltrials, ~] = signal_to_trials(subj3_control_left, stim_times, prest, post);
[crtrials, ~] = signal_to_trials(subj3_control_right, stim_times, prest, post);

time = t / Fs;

%% Normalize
norm_func = @(X) (X - mean(X, 2)) ./ std(X, 0, 2);
ltrials = norm_func(ltrials);
rtrials = norm_func(rtrials);
cltrials = norm_func(cltrials);
crtrials = norm_func(crtrials);

%% Feature Extraction - CWT
fprintf('Extracting CWT features...\n');
fb = cwtfilterbank('SignalLength', length(t), 'SamplingFrequency', Fs, ...
    'FrequencyLimits', freq_limits, 'VoicesPerOctave', voices_per_octave);
CWT_left = extract_cwt_features(ltrials, fb);
CWT_right = extract_cwt_features(rtrials, fb);
CWT_cl = extract_cwt_features(cltrials, fb);
CWT_cr = extract_cwt_features(crtrials, fb);

%% Feature Extraction - Wavelet Scattering
fprintf('Extracting WS features...\n');
sf = waveletScattering('SignalLength', target_length, 'SamplingFrequency', Fs);
[ltrials_rs, rtrials_rs, cltrials_rs, crtrials_rs] = ...
    deal(resize_trials(ltrials, target_length), resize_trials(rtrials, target_length), ...
         resize_trials(cltrials, target_length), resize_trials(crtrials, target_length));
WS_left = extract_scattering_features(sf, ltrials_rs);
WS_right = extract_scattering_features(sf, rtrials_rs);
WS_cl = extract_scattering_features(sf, cltrials_rs);
WS_cr = extract_scattering_features(sf, crtrials_rs);

%% PCA (CWT)
[CWT_X_left, y_left] = combine_features(CWT_left, CWT_cl);
[~, score_left, ~, ~, explained_left] = pca(CWT_X_left);
cumsum_left = cumsum(explained_left);
num_pca_left = find(cumsum_left >= explained_variance_threshold, 1);
CWT_left_reduced = score_left(:, 1:num_pca_left);

[CWT_X_right, y_right] = combine_features(CWT_right, CWT_cr);
[~, score_right, ~, ~, explained_right] = pca(CWT_X_right);
cumsum_right = cumsum(explained_right);
num_pca_right = find(cumsum_right >= explained_variance_threshold, 1);
CWT_right_reduced = score_right(:, 1:num_pca_right);

%% CLASSIFICATION (LEFT)
[res_left] = classify_task_gridsvm('Left Cortex', WS_left, WS_cl, CWT_left_reduced, y_left, num_top_features);

%% CLASSIFICATION (RIGHT)
[res_right] = classify_task_gridsvm('Right Cortex', WS_right, WS_cr, CWT_right_reduced, y_right, num_top_features);

%% Plot overall accuracy comparison
figure;
bar_data = [res_left.svm_ws, res_left.knn_ws; res_left.svm_cwt, res_left.knn_cwt; 
            res_right.svm_ws, res_right.knn_ws; res_right.svm_cwt, res_right.knn_cwt];
bar(bar_data*100);
set(gca, 'XTickLabel', {'Left WS', 'Left CWT', 'Right WS', 'Right CWT'});
legend('SVM', 'KNN');
ylabel('Test Accuracy (%)');
title('Classification Accuracy per Method and Hemisphere');
grid on;

%% === FUNCTIONS ===

function [X, y] = combine_features(X1, X2)
    X = [X1; X2];
    y = [zeros(size(X1,1),1); ones(size(X2,1),1)];
end

function trials_rs = resize_trials(trials, len)
    trials_rs = zeros(size(trials,1), len);
    for i = 1:size(trials,1)
        sig = trials(i,:);
        trials_rs(i,:) = [sig(1:min(end, len)), zeros(1, max(0, len - numel(sig)))];
    end
end

function CWT_features = extract_cwt_features(trials, fb)
    N = size(trials, 2);
    n_trials = size(trials,1);
    [f_len, ~] = size(cwt(trials(1,:), 'FilterBank', fb));
    CWT_features = zeros(n_trials, f_len * N);
    for i = 1:n_trials
        coeffs = cwt(trials(i,:), 'FilterBank', fb);
        CWT_features(i,:) = reshape(abs(coeffs), 1, []);
    end
end

function WS_feats = extract_scattering_features(sf, trials)
    n_trials = size(trials, 1);
    feats = featureMatrix(sf, trials', 'Transform', 'log');
    [rows, cols, ~] = size(feats);
    WS_feats = zeros(n_trials, rows * cols);
    for i = 1:n_trials
        WS_feats(i,:) = reshape(feats(:,:,i), 1, []);
    end
end

function result = classify_task_gridsvm(name, WS_stim, WS_ctrl, CWT_reduced, y, top_k)
    X = [WS_stim; WS_ctrl];
    [idx, ~] = rankfeatures(X', y', 'criterion', 'entropy');
    top_idx = idx(1:min(top_k, end));
    X_selected = X(:, top_idx);

    %% Grid Search SVM
    C_vals = 1:10;
    gamma_vals = [0.001, 0.01, 0.1, 1];
    best_acc_svm_ws = 0; best_c_ws = NaN; best_g_ws = NaN;
    best_acc_svm_cwt = 0; best_c_cwt = NaN; best_g_cwt = NaN;

    for C = C_vals
        for g = gamma_vals
            t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', C, 'KernelScale', 1/sqrt(2*g));
            acc_ws = crossval(fitcecoc(X_selected, y, 'Learners', t), 'KFold', 5);
            acc_cwt = crossval(fitcecoc(CWT_reduced, y, 'Learners', t), 'KFold', 5);
            acc_ws = 1 - kfoldLoss(acc_ws);
            acc_cwt = 1 - kfoldLoss(acc_cwt);

            if acc_ws > best_acc_svm_ws
                best_acc_svm_ws = acc_ws; best_c_ws = C; best_g_ws = g;
            end
            if acc_cwt > best_acc_svm_cwt
                best_acc_svm_cwt = acc_cwt; best_c_cwt = C; best_g_cwt = g;
            end
        end
    end

    %% KNN GridSearch
    best_acc_knn_ws = 0; best_k_ws = 1;
    best_acc_knn_cwt = 0; best_k_cwt = 1;
    for k = 1:10
        mdl_ws = fitcknn(X_selected, y, 'NumNeighbors', k);
        acc_ws = 1 - kfoldLoss(crossval(mdl_ws, 'KFold', 5));
        if acc_ws > best_acc_knn_ws
            best_acc_knn_ws = acc_ws;
            best_k_ws = k;
        end

        mdl_cwt = fitcknn(CWT_reduced, y, 'NumNeighbors', k);
        acc_cwt = 1 - kfoldLoss(crossval(mdl_cwt, 'KFold', 5));
        if acc_cwt > best_acc_knn_cwt
            best_acc_knn_cwt = acc_cwt;
            best_k_cwt = k;
        end
    end

    %% Print clean results
    fprintf('\n%s - WS-SVM Best Accuracy: %.2f%% with C=%d, gamma=%.3f\n', name, 100*best_acc_svm_ws, best_c_ws, best_g_ws);
    fprintf('%s - WS-KNN Accuracy (K=%d): %.2f%%\n', name, best_k_ws, 100*best_acc_knn_ws);
    fprintf('%s - CWT-SVM Best Accuracy: %.2f%% with C=%d, gamma=%.3f\n', name, 100*best_acc_svm_cwt, best_c_cwt, best_g_cwt);
    fprintf('%s - CWT-KNN Accuracy (K=%d): %.2f%%\n', name, best_k_cwt, 100*best_acc_knn_cwt);

    %% Return for plot
    result = struct("svm_ws", best_acc_svm_ws, "knn_ws", best_acc_knn_ws, ...
                   "svm_cwt", best_acc_svm_cwt, "knn_cwt", best_acc_knn_cwt);
end