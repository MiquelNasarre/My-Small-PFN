
## --- LICENSE AND COPYRIGHT ---
## =======================================================================================
##  * My Small PFN — a competitive proof-of concept for Prior-Data Fitted Networks
##  * Copyright (c) 2026 Miguel Nasarre Budiño
##  * Licensed under the MIT License. See LICENSE file.
## =======================================================================================

from __future__ import annotations

import torch

if __name__ == "__main__":
    from my_small_PFN import MyRegressorPFN


class FeatureEffects:
    '''
    Feature effect functions study how individual features values affect the
    output prediction, this allows for one-dimensional plots of the target
    with respect to a single feature, both on single samples or globally.

    The methods included in this class are ICE, PD and ALE, for more information
    on each function you can check their respective descriptions.
    '''

    @staticmethod
    def individual_conditional_expectation(model: MyRegressorPFN, 
            X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, 
            feature: int, grid_values: list[float]) -> torch.Tensor:
        '''
        Individual Conditional Expectation (ICE) checks how a particular feature affects
        the predicted output within a set of test samples. 

        This is done by providing a grid of values for the variable to take and running 
        the forward pass on all the test samples for all the grid values. 

        This method can be used to study how a variable modifies the output conditioned 
        by the other values, useful to find linear or non-linear relationships between
        features and targets, and heterogeneity.
        '''

        # Let's first make sure test data is a proper tensor.
        X_test  = torch.as_tensor(X_test, dtype=torch.float32) # (test_size?,  F)

        # Unsqueeze test size if necessary.
        if X_test.dim() == 1:
            X_test = X_test.unsqueeze(0) # (test_size, F)

        # Dimension values.
        K = len(grid_values)
        n_test, F = X_test.shape

        # Create the grid test.
        grid_test = X_test.unsqueeze(0).repeat(K, 1, 1).transpose(0, 1)  # (test_size, K, F)
        
        # Assign all the values to the grid.
        for k, val in enumerate(grid_values):
            grid_test[:, k, feature] = val

        # Run a forward pass on all the values.
        grid_test = grid_test.contiguous().view(n_test * K, F)  # (test_size * K, F)
        preds = model.fit(X_train, y_train).predict(grid_test, output='mean') # (test_size * K)

        # Reshape again and return predictions on all test samples for all the grid values.
        return preds.reshape([n_test, K]) # (test_size, K)

    @staticmethod
    def partial_dependence(model: MyRegressorPFN, 
            X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, 
            feature: int, grid_values: list[float]) -> torch.Tensor:
        '''
        Partial Dependence (PD) describes the overall effect of a feature on the predicted
        output. Losing locality with respect to the previous method but obtaining a global 
        image of the selected feature effect.
        
        This is done by considering a grid of values for the selected feature as well as a list 
        of training examples, then averaging the model predictions on all the test examples for 
        the different grid values.

        As described this is essentially the same as we do for the ICE function but averaging
        the output values at the end, therefore the implementation is extremely simple.
        '''

        # Get the ice output with the specified configuration
        ice_out = FeatureEffects.individual_conditional_expectation(
            model,X_train,y_train,X_test,feature,grid_values) # (test_size, K)

        # Return average over all test examples.
        return ice_out.mean(dim=0) # (K,)

    @staticmethod
    def accumulated_local_effect(model: MyRegressorPFN, 
            X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, 
            feature: int, bins: int):
        '''
        Methods like partial dependence can sometimes be misleading, since they assign
        arbitrarily chosen values to a feature that might not be possible within the 
        current feature values, making it less reliable and unstable.

        ALE solves that problem by only evaluating feature values when these are close 
        to the values found in the real data.

        This is done by bucketizing the test samples into quantiles according to that 
        feature value and evaluating the local effect on test samples inside their 
        particular bucket. Then accumulating to get the global effect.

        The downside of this method is that it requires a lot more test examples to 
        get a clean picture of the global feature effect.
        '''

        # Let's first make sure test data is a proper tensor.
        X_test  = torch.as_tensor(X_test, dtype=torch.float32) # (test_size, F)
        n_test = X_test.shape[0]
        x_f = X_test[:, feature].contiguous()

        # Create bin boundaries inside the specified range.
        edges = torch.quantile(x_f, torch.linspace(0, 1, bins + 1)) # (bins + 1,)

        # Store bin idx for each test sample
        bin_idx = torch.bucketize(x_f, edges[1:-1], right=False)  # (test_size,)

        # Build big test batch
        X_lower = X_test.clone()
        X_upper = X_test.clone()
        X_lower[:, feature] = edges[bin_idx]
        X_upper[:, feature] = edges[bin_idx + 1]

        big_batch = torch.cat([X_lower, X_upper], dim=0)  # (2 * test_size, F)

        # Run forward pass on all data points
        preds = model.fit(X_train, y_train).predict(big_batch, output='mean') # (2 * test_size,)

        preds_lower = preds[:n_test]
        preds_upper = preds[n_test:]
        delta = preds_upper - preds_lower  # (test_size,)

        # Sum per bin and average
        sum_delta = torch.zeros(bins).index_add_(0, bin_idx, delta)  # (bins,)
        counts = torch.bincount(bin_idx, minlength=bins)
        local_effects = sum_delta / counts  # (bins,)

        # Accumulate / "Integrate" and center
        ale = torch.cumsum(local_effects, dim=0)
        ale -= ale.mean()        

        # Return bin centers and ALE
        return (edges[1:] + edges[:-1]) / 2.0, ale


class FeatureImportance:
    '''
    Feature importance functions study how important individual features are to 
    the final predictions, this can be done by studying how error changes globally
    when features are removed, or by seeing how predictions change locally when using
    different combinations of features.

    These methods are particularly powerful with PFNs, since on traditional ML they 
    would require retraining for every single experiment, but with PFN this is just a 
    single forward pass.

    The methods included in this class are LOCO and kernel SHAP, for more information
    on each function you can check their respective descriptions.
    '''

    @staticmethod
    def leave_one_covariate_out(model: MyRegressorPFN, 
            X_train: torch.Tensor, y_train: torch.Tensor,
            X_test: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
        '''
        Leave One Covariate Out (LOCO) is a very simple idea, just retrain the model without one 
        of the features and see how it performs, if the model is good this gives us a good picture 
        of the importance of each feature by seeing how their absence affects the final score.

        This method is usually unviable for other types of neural networks since retraining is too
        time consuming to consider. This is not the case for PFNs, since they learn in context, they
        can evaluate a LOCO score on a single forward pass, significantly reducing computation for 
        this kind of methods.

        This function applies LOCO for all the features and returns the MSE loss difference with
        respect to the full dataset.
        '''

        # Make sure we have proper tensors.
        X_train = torch.as_tensor(X_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)
        X_test  = torch.as_tensor(X_test,  dtype=torch.float32)
        y_test  = torch.as_tensor(y_test,  dtype=torch.float32)

        n_features = X_train.shape[1]
        mse = lambda preds: torch.mean((preds.flatten() - y_test.flatten()) ** 2) 

        # Baseline model loss.
        baseline_loss = mse(model.fit(X_train, y_train).predict(X_test, output='mean'))

        # Tensor to store scores.
        loco_scores = torch.zeros(n_features)

        # Iterate through features.
        for j in range(n_features):

            # Strip the feature out.
            X_train_reduced = torch.cat([X_train[:, :j], X_train[:, j+1:]], dim=1)
            X_test_reduced = torch.cat([X_test[:, :j], X_test[:, j+1:]], dim=1)

            # Compute loss difference.
            reduced_loss = mse(model.fit(X_train_reduced, y_train).predict(X_test_reduced, output='mean'))
            loco_scores[j] = reduced_loss - baseline_loss

        # Return scores.
        return loco_scores

    @staticmethod
    def kernel_shap(model: MyRegressorPFN, 
            X_train: torch.Tensor, y_train: torch.Tensor,
            X_test: torch.Tensor, n_subsets: int = 256, include_empty_full: bool = True) -> torch.Tensor:
        '''
        For any given test case the idea behind kernel SHAP is to break down the final prediction
        into single feature contributions, allowing to know how much each feature added to the final
        output.

        This is done by considering all possible feature combinations with and without a given feature
        and then performing a weighted sum of the differences.

        Since this method is quite computationally demanding there is an approximation to simplify it. 
        If you assume the sum equation is true it holds that each feature contributes a certain value 
        to the prediction, therefore this can be solved linearly.

        First you compute a smaller amount of feature subsets, then you solve a system assuming each 
        feature has the same contribution on all subsets where it is present. The system is weighted
        by the Shapley kernel. The solution to the system gives you each feature contribution to that 
        particular test case.
        '''

        # Make sure we have proper tensors.
        X_train = torch.as_tensor(X_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)
        X_test  = torch.as_tensor(X_test,  dtype=torch.float32)

        # Unsqueeze test size if necessary.
        if X_test.dim() == 1:
            X_test = X_test.unsqueeze(0) # (test_size, F)

        # Dimension values.
        n_test, F = X_test.shape

        # Generate arbitrary subsets.
        subsets = torch.randint(low=0, high=2, size=[n_subsets, F]).bool() # (n_subsets, F)

        # If empty and full subsets are included enforce them.
        if include_empty_full:
            subsets[0, :] = False
            subsets[1, :] = True

        # Tensor to store shapley values.
        values = torch.zeros([n_subsets, n_test]) # (n_subsets, test_size)

        # Iterate through all subsets.
        for m in range(n_subsets):
            # Select subset.
            S = subsets[m] # (F,)

            # Prepare sets.
            X_train_S = X_train[:, S] # (train_size, |S|)
            x_test_S  =  X_test[:, S] # ( test_size, |S|)

            # Run forward pass. (Unsqueeze for shape compatibility if F == 0)
            values[m] = model.fit(X_train_S.unsqueeze(0), y_train.unsqueeze(0)).predict(x_test_S.unsqueeze(0), output='mean') # (test_size,)

        # Design matrix A, mask of feature subsets for the solution, contains bias row.
        A = torch.cat([torch.ones(n_subsets, 1), subsets.float()], dim=1)  # (n_subsets, F + 1)

        # Weights per subset row
        k = subsets.sum(dim=1).to(torch.float32)  # (n_subsets,)

        # Compute Shapley kernel.
        logC = (torch.lgamma(torch.tensor(F + 1.0)) -
                torch.lgamma(k + 1.0) -
                torch.lgamma(torch.tensor(F + 1.0) - k))

        C = torch.exp(logC)  # (n_subsets,)
        w = torch.full([n_subsets,], 1e4, dtype=torch.float32)
        middle = (k > 0) & (k < F)
        w[middle] = (F - 1.0) / (C[middle] * k[middle] * (F - k[middle]))
        sqrtw = torch.sqrt(w).unsqueeze(1)  # (n_subsets, 1)

        # Get system vectors for all test samples.
        Aw = A * sqrtw      # (n_subsets, F + 1)
        Bw = values * sqrtw # (n_subsets, test_size)

        # Solve weighted least squares for all test samples.
        sol = torch.linalg.lstsq(Aw, Bw).solution # (F + 1, test_size)

        # Extract phi and return
        phi = sol[1:].T

        # Squeeze if only one test case
        if n_test == 1:
            return phi.squeeze(0) # (F,)

        return phi # (test_size, F)


class DataValuation:
    '''
    Data valuation functions study how individual training cases affect the models
    predictions. Usually measuring changes on test set loss given different training
    set configurations.

    These methods are particularly powerful with PFNs, since on traditional ML they 
    would require retraining for every single experiment, but with PFN this is just a 
    single forward pass.

    The methods included in this class are LOO and data Shapley, for more information
    on each function you can check their respective descriptions.
    '''

    @staticmethod
    def leave_one_out(model: MyRegressorPFN,
            X_train: torch.Tensor, y_train: torch.Tensor,
            X_test: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
        '''
        Leave One Out (LOO) is a simple concept, how does one single training example affect the output
        MSE? This is a very helpful question since it helps us guess which data is helpful and which data
        is mostly noise.

        In a regular ML setting this is usually very time consuming since it implies retraining the model 
        without a single training example for each one you have. With PFNs though this becomes trivial,
        since it only requires doing a forward pass with the missing example.

        This function does that for each training example and returns the MSE loss difference for each 
        case, which can easily be used for fine tuning with the input data.
        '''

        # Make sure we have proper tensors.
        X_train = torch.as_tensor(X_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)
        X_test  = torch.as_tensor(X_test,  dtype=torch.float32)
        y_test  = torch.as_tensor(y_test,  dtype=torch.float32)
        
        # Store sizes
        train_size, F = X_train.shape

        # MSE function
        mse = lambda preds: torch.mean((preds.flatten() - y_test.flatten()) ** 2) 

        # Variable to store individual scores
        default_score = mse(model.fit(X_train, y_train).predict(X_test, output='mean')).item()
        scores = torch.full([train_size], -default_score)

        for i in range(train_size):
            # Eliminate train sample i
            X_i_train = torch.cat([X_train[:i], X_train[i+1:]], dim=0)
            y_i_train = torch.cat([y_train[:i], y_train[i+1:]], dim=0)
            # Compute MSE and add
            scores[i] += mse(model.fit(X_i_train, y_i_train).predict(X_test, output='mean'))

        # Return scores
        return scores

    @staticmethod
    def data_shapley(model: MyRegressorPFN,
            X_train: torch.Tensor, y_train: torch.Tensor,
            X_test: torch.Tensor, y_test: torch.Tensor,
            n_permutations: int = 32) -> torch.Tensor:
        '''
        Data Shapley is the logical continuation of LOO, instead of simply computing the difference 
        it makes taking a training example from the full set, you consider all possible training set 
        configurations and see how each sample contributes on average across all of those configurations.

        Similar to kernel SHAP this would require too many computations, therefore some approximations
        need to be done to make this method viable. In this case a random permutation is done to the 
        training examples and these are added one by one, the difference in MSE is considered its 
        contribution, then that difference is averaged.

        This function returns that average over all the training examples, giving us a similar result 
        to LOO but theoretically with more stable results.
        '''

        # Make sure we have proper tensors.
        X_train = torch.as_tensor(X_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)
        X_test  = torch.as_tensor(X_test,  dtype=torch.float32)
        y_test  = torch.as_tensor(y_test,  dtype=torch.float32)

        # Store sizes
        n_train, F = X_train.shape

        # Define MSE function and initial loss value.
        mse = lambda preds: torch.mean((preds.flatten() - y_test.flatten()) ** 2)
        empty_value = -mse(model.fit(X_train[:0], y_train[:0]).predict(X_test, output='mean')).item()

        # Vector to store MSE difference values
        shapley = torch.zeros(n_train)

        # Do n_permutation times
        for _ in range(n_permutations):
            # Choose a random permutation
            perm = torch.randperm(n_train)

            # Initialize previous value
            prev_value = empty_value

            for i in range(n_train):
                # Fit up to training example i in the permutation and predict
                value = -mse(model.fit(X_train[perm[:i+1]], y_train[perm[:i+1]]).predict(X_test, output='mean')).item()

                # Add error difference to the corresponding Shapley
                shapley[perm[i]] += value - prev_value

                # Store new value as previous
                prev_value = value

        # Return average error difference
        return shapley / n_permutations
