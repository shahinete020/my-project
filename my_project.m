%% =========================================================
%  PREDICTIVE MAINTENANCE SYSTEM FOR MOTORS
%  Complete MATLAB Project in One File
%
%  Features:
%  1. Generates synthetic motor sensor dataset
%  2. Trains Machine Learning model
%  3. Tests accuracy
%  4. Displays confusion matrix
%  5. Predicts motor condition from user input
%  6. Visualizes sensor data
%
%  Software Required:
%  MATLAB + Statistics and Machine Learning Toolbox
%
%  Author: ChatGPT
%% =========================================================

clc;
clear;
close all;

disp('==========================================');
disp(' PREDICTIVE MAINTENANCE SYSTEM FOR MOTORS ');
disp('==========================================');

%% =========================================================
% STEP 1: GENERATE SYNTHETIC DATASET
%% =========================================================

disp('Generating Dataset...');

n = 1000;

% ---------------------------
% HEALTHY MOTOR DATA
% ---------------------------
healthy_vibration = randn(n/2,1)*0.2 + 1;
healthy_temperature = randn(n/2,1)*2 + 40;
healthy_current = randn(n/2,1)*0.5 + 5;

% ---------------------------
% FAULTY MOTOR DATA
% ---------------------------
faulty_vibration = randn(n/2,1)*0.5 + 4;
faulty_temperature = randn(n/2,1)*5 + 80;
faulty_current = randn(n/2,1)*1 + 10;

% Combine Features
vibration = [healthy_vibration; faulty_vibration];
temperature = [healthy_temperature; faulty_temperature];
current = [healthy_current; faulty_current];

% Labels
condition = [zeros(n/2,1); ones(n/2,1)];
% 0 = Healthy
% 1 = Faulty

% Create Table
motor_data = table(vibration, temperature, current, condition);

disp('Dataset Generated Successfully');

%% =========================================================
% STEP 2: VISUALIZE DATA
%% =========================================================

figure('Name','Motor Sensor Data');

subplot(3,1,1);
scatter(1:n, vibration, 15, condition, 'filled');
title('Vibration Data');
xlabel('Sample');
ylabel('Vibration');

subplot(3,1,2);
scatter(1:n, temperature, 15, condition, 'filled');
title('Temperature Data');
xlabel('Sample');
ylabel('Temperature');

subplot(3,1,3);
scatter(1:n, current, 15, condition, 'filled');
title('Current Data');
xlabel('Sample');
ylabel('Current');

sgtitle('Motor Condition Visualization');

%% =========================================================
% STEP 3: PREPARE TRAINING DATA
%% =========================================================

disp('Preparing Training Data...');

X = motor_data{:,1:3};
Y = motor_data.condition;

% Split Data
cv = cvpartition(Y,'HoldOut',0.2);

Xtrain = X(training(cv),:);
Ytrain = Y(training(cv));

Xtest = X(test(cv),:);
Ytest = Y(test(cv));

%% =========================================================
% STEP 4: TRAIN MACHINE LEARNING MODEL
%% =========================================================

disp('Training Machine Learning Model...');

model = fitctree(Xtrain, Ytrain);

disp('Model Training Complete');

%% =========================================================
% STEP 5: TEST MODEL
%% =========================================================

disp('Testing Model...');

Ypred = predict(model, Xtest);

accuracy = sum(Ypred == Ytest)/length(Ytest)*100;

fprintf('\n====================================\n');
fprintf(' MODEL ACCURACY = %.2f%%\n', accuracy);
fprintf('====================================\n');

%% =========================================================
% STEP 6: CONFUSION MATRIX
%% =========================================================

figure('Name','Confusion Matrix');
confusionchart(Ytest, Ypred);

title('Motor Fault Classification');

%% =========================================================
% STEP 7: FEATURE IMPORTANCE
%% =========================================================

figure('Name','Feature Importance');

importance = predictorImportance(model);

bar(importance);

xticklabels({'Vibration','Temperature','Current'});

ylabel('Importance Score');

title('Feature Importance Analysis');

grid on;

%% =========================================================
% STEP 8: REAL-TIME USER PREDICTION
%% =========================================================

disp(' ');
disp('====================================');
disp(' REAL-TIME MOTOR HEALTH PREDICTION ');
disp('====================================');

while true

    vibration_input = input('Enter Vibration Value: ');
    temperature_input = input('Enter Temperature Value: ');
    current_input = input('Enter Current Value: ');

    sample = [vibration_input temperature_input current_input];

    prediction = predict(model, sample);

    fprintf('\n');

    if prediction == 0
        disp('--------------------------------');
        disp(' MOTOR STATUS: HEALTHY');
        disp('--------------------------------');
    else
        disp('--------------------------------');
        disp(' MOTOR STATUS: FAULTY');
        disp('--------------------------------');
    end

    fprintf('\n');

    choice = input('Check another motor? (y/n): ','s');

    if lower(choice) ~= 'y'
        break;
    end

end

%% =========================================================
% STEP 9: SAVE TRAINED MODEL
%% =========================================================

save('PredictiveMaintenanceModel.mat','model');

disp(' ');
disp('Model Saved Successfully');
disp('File Name: PredictiveMaintenanceModel.mat');

%% =========================================================
% STEP 10: FINAL MESSAGE
%% =========================================================

disp(' ');
disp('====================================');
disp(' PROJECT COMPLETED SUCCESSFULLY ');
disp('====================================');