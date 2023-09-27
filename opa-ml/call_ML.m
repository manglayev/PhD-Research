%% clear
clear ; close all; clc;

UEs = [2 3 4 5 6 7 8 9 10 11 12];
%UEs = 3;
linearRegressionTime = zeros(1, length(UEs));
file_gradient_time = 'Article_6_7/results_new/gradient_time.txt';
file_normal_time = 'Article_6_7/results_new/normal_time.txt';

timeGradientArray = zeros(1, length(UEs));
timeNormalArray = zeros(1, length(UEs));

for a = 1:1:length(UEs)
    [timeGradient, timeNormal, gradient_coefficients, normal_coefficients] = Linear_Regression(UEs(a));
    %% write time to predict
    timeGradientArray(a) = timeGradient;
    timeNormalArray(a) = timeNormal;
    %% write predicted gradient power allocation coefficients to files
    file_power_gradient = strcat('Article_6_7/results_new/gradient_coefficients_', num2str(UEs(a)));
    file_power_gradient = strcat(file_power_gradient,'.txt');
    writematrix(gradient_coefficients, file_power_gradient, 'Delimiter',' ');
    %% write predicted normal power allocation coefficients to files
    file_power_normal = strcat('Article_6_7/results_new/normal_coefficients_', num2str(UEs(a)));
    file_power_normal = strcat(file_power_normal,'.txt');
    writematrix(normal_coefficients, file_power_normal, 'Delimiter',' ');
end

writematrix(timeGradientArray, file_gradient_time, 'Delimiter',' ');
writematrix(timeNormalArray, file_normal_time, 'Delimiter',' ');