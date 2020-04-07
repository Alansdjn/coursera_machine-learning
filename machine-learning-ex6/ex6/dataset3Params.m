function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
prediction_err = -1;

for i = 1 : length(C_vec)
	C_tmp = C_vec(i);
	for j = 1 : length(sigma_vec)
		fprintf('Training i = %d, j = %d ... \n', i, j);
		sigma_tmp = sigma_vec(j);
		model = svmTrain(X, y, C_tmp, @(x1, x2) gaussianKernel(x1, x2, sigma_tmp));
		predictions = svmPredict(model, Xval);
		prediction_err_ij = mean(double(predictions ~= yval));
		if prediction_err == -1
			prediction_err = prediction_err_ij;
		end
		if prediction_err >= prediction_err_ij
			prediction_err = prediction_err_ij;
			C = C_tmp;
			sigma = sigma_tmp;
		end
		%fprintf('prediction_err_%d%d = %d\n', i, j, prediction_err_ij);
		%fprintf('size(prediction_err_%d%d) = %d\n', i, j, size(prediction_err_ij));
		%prediction_err(i,j) = prediction_err_ij;
		%fprintf('C = %f, sigma = %f, model.b = %f, model.w = %f\n', C_tmp, sigma_tmp, model.b, model.w);
		%fprintf('C = %f, sigma = %f, model.b = %f, model.w = %f\n', C_tmp, sigma_tmp, model.b, model.w);
	end
end

fprintf('C = %f, sigma = %f\n', C, sigma);







% =========================================================================

end
