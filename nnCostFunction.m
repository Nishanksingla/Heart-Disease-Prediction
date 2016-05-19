function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1) X];

Z2 = X*Theta1';
A2 = sigmoid(Z2);
A2 = [ones(m,1) A2];

Z3 = A2*Theta2';
A3 = sigmoid(Z3);
h_theta = A3; 

Y = zeros(m,num_labels); 
for i=1:m,
  Y(i,y(i))=1;
end

J = 1/m * sum (sum ( (-Y) .* log(h_theta) - (1-Y) .* log(1-h_theta)));

t1=Theta1(:,2:size(Theta1,2));
t2=Theta2(:,2:size(Theta2,2));

Reg = (lambda/(2*m)) *(sum(sum(t1.^2)) + sum(sum(t2.^2)));

J = J + Reg;

for i=1:m
    a1=X(i,:);
    z2 = Theta1 * a1';
    
    a2 = sigmoid(z2);
    a2 = [1; a2];

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    z2 = [1; z2];
    
    delta3 = a3 - Y(i,:)';
    delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);
    
    delta2=delta2(2:end);
    Theta2_grad = Theta2_grad + delta3 * a2';
    Theta1_grad = Theta1_grad + delta2 * a1;

end

Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
	
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));
	
	
Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
