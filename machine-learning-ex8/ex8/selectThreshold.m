function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

%printf("max(pval) == %f\n", max(pval));
%printf("min(pval) == %f\n", min(pval));

%for i = 1:size(yval)
%    printf("pval(%d) = %f, yval(%d) = %f\n", i, pval(i), i, yval(i));
%end

stepsize = (max(pval) - min(pval)) / 1000;
%printf("stepsize == %f\n", stepsize);
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    %printf("epsilon == %f\n", epsilon);

    cvPredictions = (pval < epsilon);
    %for i = 1:size(yval)
    %    printf("cvPredictions(%d) = %d, yval(%d) = %d\n", i, cvPredictions(i), i, yval(i));
    %end

    tp = sum((cvPredictions == 1) & (yval == 1));
    fp = sum((cvPredictions == 1) & (yval == 0));
    fn = sum((cvPredictions == 0) & (yval == 1));

    prec = tp / (tp + fp);
    rec = tp / (tp + fn);

    F1 = (2 * prec * rec) / (prec + rec);
    %printf("epsilon = %f, tp = %d, fp = %d, fn = %d\n", epsilon, tp, fp, fn);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
    %printf("epsilon = %f, F1 = %f, bestF1 = %f, bestEpsilon = %f\n", epsilon, F1, bestF1, bestEpsilon);
end

end
