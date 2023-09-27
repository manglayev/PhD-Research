%% ML Linear Regression 682 rows features equal to number of UEs
function [timeGradient, timeNormal, gradient_coefficients, normal_coefficients] = Linear_Regression(NumberOfUsers)
    %% initialize
    distances = load('Article_6_7/data/distances_all.txt');
    fileCoefficients = strcat('Article_6_7/data/ES_coefficients_', num2str(NumberOfUsers));
    fileCoefficients = strcat(fileCoefficients,'.txt');
    
    coefficients = dlmread(fileCoefficients,' ');    

    X = distances(:, 1:NumberOfUsers);
    y = coefficients(:, 1:NumberOfUsers);
    m = length(y);
    
    mean_distance = [178 200 222 242 260 279 297 314 330 345 357 372];
    
    [X, mu, sigma] = featureNormalize(X);
    % Add intercept term to X
    X = [ones(m, 1) X];
    alpha = 0.001;
    num_iters = 10000;
    
    vectorOfPrediction = mean_distance(1:NumberOfUsers);
    vectorOfPrediction = (vectorOfPrediction-mu)./sigma;
    vectorOfPrediction = [ones(1, 1) vectorOfPrediction];
    
    %% for UEs in a range(2:NumberOfUsers)
    %Init Theta and Run gradient descent 
    theta_1 = zeros(size(X,2), 1);
    gradient_coefficients = zeros(1, NumberOfUsers);
    timeGradient = 0;
%   tic
%   for a = 1:1:NumberOfUsers
%       theta_1 = gradientDescentMulti(X, y(:, a), theta_1, alpha, num_iters);
%       gradient_coefficients(a) = round(vectorOfPrediction*theta_1, 2);
%   end
%   timeGradient = timeGradient + toc;
%   timeGradient = (timeGradient*1000);
%   Run normal equation 
    timeNormal = 0;
    normal_coefficients = zeros(1, NumberOfUsers);
    
     for iteration = 1:1:100000
         tic    
         for b = 1:1:NumberOfUsers
             theta_2 = normalEqn(X, y(:, b));
             normal_coefficients(b) = round(vectorOfPrediction*theta_2, 2);
         end
         timeNormal = timeNormal + toc;
     end
     
     timeNormal = timeNormal/100000;
     timeNormal = round((timeNormal*1000),4);
    %fprintf('gradient: %.4f \n', timeGradient);
    %fprintf('normal: %.4f \n', timeNormal);
end