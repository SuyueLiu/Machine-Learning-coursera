function p = predict(Theta1, Theta2, a1)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(a1, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(a1, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(m,1),a1]; % 补充X0, (5000,401)

z2 = Theta1 * a1.'; % 隐藏层的输入
z2 = sigmoid(z2); % 隐藏层的输出, (25,5000)
a2 =[ones(1,m); z2]; % 对隐藏层的输入增加一个值为1的偏置,(25,5000)
% 隐藏层先过激活函数，再增加偏置

y_pred = sigmoid(Theta2 * a2); % (10,5000)
y_pred = y_pred.' ; %(5000,10)

[~, p] = max(y_pred, [], 2); % (5000,1),返回最大值和最大值的index，保留index






% =========================================================================


end
