%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% [X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

alpha = 0.1;
num_iters = 400;

theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

theta